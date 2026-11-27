#include "engine/voxel/Mesher.hpp"

#include "engine/core/Profile.hpp"

#include <array>
#include <bit>

namespace engine::voxel {

namespace {

struct Offset {
    i32 dx, dy, dz;
};

// Indexed by Face.
constexpr std::array<Offset, kFaceCount> kFaceOffsets{{
    {+1, 0, 0},
    {-1, 0, 0},
    {0, +1, 0},
    {0, -1, 0},
    {0, 0, +1},
    {0, 0, -1},
}};

// Index stride of one step along each axis in the chunk's block array.
constexpr std::array<u32, 3> kAxisStride{1, kChunkArea, kChunkSize}; // x, y, z

// Occupancy bitmasks: one u32 per column of 32 voxels along an axis.
//   x[y*32 + z] bit i -> voxel (i, y, z)
//   y[z*32 + x] bit i -> voxel (x, i, z)
//   z[y*32 + x] bit i -> voxel (x, y, i)
// For each axis, the column index (a*32 + b) uses the same (row, col) order
// as that axis' BoundaryMasks, so neighbour bits line up directly.
struct Columns {
    std::array<u32, kChunkArea> x;
    std::array<u32, kChunkArea> y;
    std::array<u32, kChunkArea> z;
};

// The two in-plane axes of each normal axis' columns: column (a, b).
constexpr std::array<std::array<u8, 2>, 3> kColumnAxes{{
    {1, 2}, // x columns: a = y, b = z
    {2, 0}, // y columns: a = z, b = x
    {1, 0}, // z columns: a = y, b = x
}};

void buildColumns(const Chunk& chunk, Columns& cols) noexcept
{
    ENGINE_PROFILE_SCOPE("buildColumns");
    cols.y.fill(0);
    cols.z.fill(0);
    const BlockId* ENGINE_RESTRICT blocks = chunk.blocks().data();

    for (u32 y = 0; y < kChunkSize; ++y) {
        for (u32 z = 0; z < kChunkSize; ++z) {
            const BlockId* row = blocks + voxelIndex(0, y, z);
            u32 xBits = 0;
            // Branchless: for each voxel, OR its solid bit into all three columns.
            // The inner loop has no data-dependent branches and contiguous
            // loads/stores, which lets the compiler vectorise it.
            for (u32 x = 0; x < kChunkSize; ++x) {
                const u32 solid = row[x] != kAir ? 1u : 0u;
                xBits |= solid << x;
                cols.y[z * kChunkSize + x] |= solid << y;
                cols.z[y * kChunkSize + x] |= solid << z;
            }
            cols.x[y * kChunkSize + z] = xBits;
        }
    }
}

[[nodiscard]] constexpr const std::array<u32, kChunkArea>& columnsFor(const Columns& cols, u32 axis) noexcept
{
    return axis == 0 ? cols.x : axis == 1 ? cols.y : cols.z;
}

// Visible faces of one column: solid voxels whose neighbour along the face
// normal is air. Inside the chunk the neighbour is the column shifted by one;
// at the border it is the bit from the neighbouring chunk's boundary mask.
template <Face F>
[[nodiscard]] ENGINE_FORCEINLINE u32 visibleFaces(const Columns& cols, const BoundaryMasks& nb, u32 a, u32 b) noexcept
{
    constexpr u32 axis = kFaceAxes[static_cast<usize>(F)].normal;
    constexpr bool positive = (static_cast<u32>(F) & 1u) == 0;
    const u32 col = columnsFor(cols, axis)[a * kChunkSize + b];
    const u32 outside = (nb.rows[static_cast<usize>(F)][a] >> b) & 1u;
    if constexpr (positive) {
        return col & ~((col >> 1) | (outside << 31));
    } else {
        return col & ~((col << 1) | outside);
    }
}

template <Face F>
[[nodiscard]] u32 countFaces(const Columns& cols, const BoundaryMasks& nb) noexcept
{
    u32 count = 0;
    for (u32 a = 0; a < kChunkSize; ++a) {
        for (u32 b = 0; b < kChunkSize; ++b) {
            count += static_cast<u32>(std::popcount(visibleFaces<F>(cols, nb, a, b)));
        }
    }
    return count;
}

template <Face F>
void emitUnitQuads(const Columns& cols, const BoundaryMasks& nb, const BlockId* ENGINE_RESTRICT blocks,
                   PackedQuad*& dst) noexcept
{
    constexpr u32 axis = kFaceAxes[static_cast<usize>(F)].normal;
    constexpr u32 axisA = kColumnAxes[axis][0];
    constexpr u32 axisB = kColumnAxes[axis][1];
    for (u32 a = 0; a < kChunkSize; ++a) {
        for (u32 b = 0; b < kChunkSize; ++b) {
            u32 mask = visibleFaces<F>(cols, nb, a, b);
            while (mask != 0) {
                const u32 i = static_cast<u32>(std::countr_zero(mask));
                mask &= mask - 1; // clear lowest set bit
                std::array<u32, 3> c{};
                c[axis] = i;
                c[axisA] = a;
                c[axisB] = b;
                *dst++ = PackedQuad::make(c[0], c[1], c[2], F, blocks[voxelIndex(c[0], c[1], c[2])]);
            }
        }
    }
}

// Greedy meshing for one face direction.
//
// 1. Scatter every visible face into a 2D bit-plane per slice along the
//    normal: plane[slice][v] bit u. Each plane row is 32 faces.
// 2. Per row, take the lowest set bit, grow it along U while the bits are set
//    and the block matches, then grow the run along V while the next row has
//    the whole run set with the same block. Clear it and emit one quad.
template <Face F>
void emitGreedyQuads(const Columns& cols, const BoundaryMasks& nb, const BlockId* ENGINE_RESTRICT blocks,
                     std::vector<PackedQuad>& out)
{
    constexpr FaceAxes axes = kFaceAxes[static_cast<usize>(F)];
    constexpr u32 axisA = kColumnAxes[axes.normal][0];
    constexpr u32 axisB = kColumnAxes[axes.normal][1];
    constexpr u32 strideN = kAxisStride[axes.normal];
    constexpr u32 strideU = kAxisStride[axes.u];
    constexpr u32 strideV = kAxisStride[axes.v];

    std::array<std::array<u32, kChunkSize>, kChunkSize> planes{}; // [slice][v], 4 KiB

    {
        ENGINE_PROFILE_SCOPE("scatterPlanes");
        for (u32 a = 0; a < kChunkSize; ++a) {
            for (u32 b = 0; b < kChunkSize; ++b) {
                u32 mask = visibleFaces<F>(cols, nb, a, b);
                // (a, b) are the column's in-plane coordinates; map them to (u, v).
                std::array<u32, 3> c{};
                c[axisA] = a;
                c[axisB] = b;
                const u32 uBit = 1u << c[axes.u];
                const u32 v = c[axes.v];
                while (mask != 0) {
                    const u32 slice = static_cast<u32>(std::countr_zero(mask));
                    mask &= mask - 1;
                    planes[slice][v] |= uBit;
                }
            }
        }
    }

    ENGINE_PROFILE_SCOPE("mergeQuads");
    for (u32 slice = 0; slice < kChunkSize; ++slice) {
        auto& rows = planes[slice];
        for (u32 v = 0; v < kChunkSize; ++v) {
            while (rows[v] != 0) {
                const u32 row = rows[v];
                const u32 u0 = static_cast<u32>(std::countr_zero(row));
                const u32 base = slice * strideN + u0 * strideU + v * strideV;
                const BlockId block = blocks[base];

                // Grow along U: contiguous set bits with the same block.
                const u32 runLength = static_cast<u32>(std::countr_one(row >> u0));
                u32 width = 1;
                while (width < runLength && blocks[base + width * strideU] == block) {
                    ++width;
                }
                const u32 runMask = (width == 32 ? ~0u : ((1u << width) - 1u)) << u0;

                // Grow along V: next row must contain the whole run, all the same block.
                u32 height = 1;
                while (v + height < kChunkSize && (rows[v + height] & runMask) == runMask) {
                    const u32 rowBase = base + height * strideV;
                    bool same = true;
                    for (u32 k = 0; k < width; ++k) {
                        same &= blocks[rowBase + k * strideU] == block;
                    }
                    if (!same) {
                        break;
                    }
                    ++height;
                }

                for (u32 k = 0; k < height; ++k) {
                    rows[v + k] &= ~runMask;
                }

                std::array<u32, 3> c{};
                c[axes.normal] = slice;
                c[axes.u] = u0;
                c[axes.v] = v;
                out.push_back(PackedQuad::make(c[0], c[1], c[2], F, block, width, height));
            }
        }
    }
}

} // namespace

BoundaryMasks buildBoundaryMasks(const ChunkNeighbours& neighbours)
{
    ENGINE_PROFILE_FUNCTION();
    BoundaryMasks masks;
    constexpr u32 last = kChunkSize - 1;
    for (u32 f = 0; f < kFaceCount; ++f) {
        const Chunk* n = neighbours.chunks[f];
        if (n == nullptr) {
            continue; // air
        }
        // The neighbour's layer touching this chunk: its voxel at `layer` along
        // the face normal. Each mask bit (row, col) is one voxel of that layer.
        const u32 normal = kFaceAxes[f].normal;
        const u32 layer = (f % 2 == 0) ? 0 : last; // +X neighbour touches at its x = 0
        const u32 rowAxis = kColumnAxes[normal][0];
        const u32 colAxis = kColumnAxes[normal][1];
        const BlockId* ENGINE_RESTRICT blocks = n->blocks().data();
        const u32 base = layer * kAxisStride[normal];
        const u32 rowStride = kAxisStride[rowAxis];
        const u32 colStride = kAxisStride[colAxis];
        auto& rows = masks.rows[f];
        for (u32 row = 0; row < kChunkSize; ++row) {
            const BlockId* line = blocks + base + row * rowStride;
            u32 bits = 0;
            for (u32 col = 0; col < kChunkSize; ++col) {
                bits |= (line[col * colStride] != kAir ? 1u : 0u) << col;
            }
            rows[row] = bits;
        }
    }
    return masks;
}

void meshChunkNaive(const Chunk& chunk, const ChunkNeighbours& neighbours, std::vector<PackedQuad>& out)
{
    ENGINE_PROFILE_FUNCTION();
    out.clear();
    constexpr i32 mask = static_cast<i32>(kChunkSize) - 1;

    for (u32 y = 0; y < kChunkSize; ++y) {
        for (u32 z = 0; z < kChunkSize; ++z) {
            for (u32 x = 0; x < kChunkSize; ++x) {
                const BlockId block = chunk.get(x, y, z);
                if (block == kAir) {
                    continue;
                }
                for (u32 f = 0; f < kFaceOffsets.size(); ++f) {
                    const auto [dx, dy, dz] = kFaceOffsets[f];
                    const i32 nx = static_cast<i32>(x) + dx;
                    const i32 ny = static_cast<i32>(y) + dy;
                    const i32 nz = static_cast<i32>(z) + dz;
                    // Outside this chunk, look in the neighbour (coordinates wrap).
                    const Chunk* src = inChunk(nx, ny, nz) ? &chunk : neighbours.chunks[f];
                    const bool neighbourSolid =
                        src != nullptr && src->get(static_cast<u32>(nx & mask), static_cast<u32>(ny & mask),
                                                   static_cast<u32>(nz & mask)) != kAir;
                    if (!neighbourSolid) {
                        out.push_back(PackedQuad::make(x, y, z, static_cast<Face>(f), block));
                    }
                }
            }
        }
    }
}

void meshChunkBinary(const Chunk& chunk, const BoundaryMasks& neighbours, std::vector<PackedQuad>& out)
{
    ENGINE_PROFILE_FUNCTION();

    Columns cols; // 12 KiB, intentionally uninitialised (buildColumns writes it)
    buildColumns(chunk, cols);

    // Pass 1: count faces so the output is sized exactly once. Popcount is a
    // single instruction, so this costs far less than per-face push_back checks.
    const u32 count = countFaces<Face::PosX>(cols, neighbours) + countFaces<Face::NegX>(cols, neighbours) +
                      countFaces<Face::PosY>(cols, neighbours) + countFaces<Face::NegY>(cols, neighbours) +
                      countFaces<Face::PosZ>(cols, neighbours) + countFaces<Face::NegZ>(cols, neighbours);
    out.resize(count);

    // Pass 2: walk set bits and write quads straight into the buffer.
    ENGINE_PROFILE_SCOPE("emitQuads");
    const BlockId* blocks = chunk.blocks().data();
    PackedQuad* dst = out.data();
    emitUnitQuads<Face::PosX>(cols, neighbours, blocks, dst);
    emitUnitQuads<Face::NegX>(cols, neighbours, blocks, dst);
    emitUnitQuads<Face::PosY>(cols, neighbours, blocks, dst);
    emitUnitQuads<Face::NegY>(cols, neighbours, blocks, dst);
    emitUnitQuads<Face::PosZ>(cols, neighbours, blocks, dst);
    emitUnitQuads<Face::NegZ>(cols, neighbours, blocks, dst);
    ENGINE_ASSERT(dst == out.data() + out.size());
}

void meshChunkGreedy(const Chunk& chunk, const BoundaryMasks& neighbours, std::vector<PackedQuad>& out)
{
    ENGINE_PROFILE_FUNCTION();
    out.clear();

    Columns cols;
    buildColumns(chunk, cols);

    const BlockId* blocks = chunk.blocks().data();
    emitGreedyQuads<Face::PosX>(cols, neighbours, blocks, out);
    emitGreedyQuads<Face::NegX>(cols, neighbours, blocks, out);
    emitGreedyQuads<Face::PosY>(cols, neighbours, blocks, out);
    emitGreedyQuads<Face::NegY>(cols, neighbours, blocks, out);
    emitGreedyQuads<Face::PosZ>(cols, neighbours, blocks, out);
    emitGreedyQuads<Face::NegZ>(cols, neighbours, blocks, out);
}

} // namespace engine::voxel
