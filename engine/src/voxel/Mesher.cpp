#include "engine/voxel/Mesher.hpp"

#include "engine/core/Bits.hpp"
#include "engine/core/Profile.hpp"

#include <algorithm>
#include <array>
#include <bit>

#if defined(__AVX2__)
    #include <immintrin.h>
#endif

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

// Solid bits of 32 consecutive voxels along x (bit i = row[i] is not air).
[[nodiscard]] ENGINE_FORCEINLINE u32 solidMask32(const BlockId* row) noexcept
{
#if defined(__AVX2__)
    // Explicit AVX2 rather than trusting auto-vectorisation (MSVC does not
    // vectorise the scalar loop below). 16 voxels per compare.
    const __m256i zero = _mm256_setzero_si256();
    const __m256i airLo = _mm256_cmpeq_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(row)), zero);
    const __m256i airHi = _mm256_cmpeq_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + 16)), zero);
    // Narrow 16-bit lanes to bytes. packs works per 128-bit half, giving
    // qwords [lo 0-7, hi 0-7, lo 8-15, hi 8-15]; reorder to 0,2,1,3.
    const __m256i packed = _mm256_permute4x64_epi64(_mm256_packs_epi16(airLo, airHi), 0xD8);
    return ~static_cast<u32>(_mm256_movemask_epi8(packed));
#else
    u32 bits = 0;
    for (u32 x = 0; x < kChunkSize; ++x) {
        bits |= (row[x] != kAir ? 1u : 0u) << x;
    }
    return bits;
#endif
}

void buildColumns(const Chunk& chunk, Columns& cols) noexcept
{
    ENGINE_PROFILE_SCOPE("buildColumns");
    const BlockId* blocks = chunk.blocks().data();

    // x columns straight from memory: each 32-voxel row along x is one mask.
    for (u32 y = 0; y < kChunkSize; ++y) {
        for (u32 z = 0; z < kChunkSize; ++z) {
            cols.x[y * kChunkSize + z] = solidMask32(blocks + voxelIndex(0, y, z));
        }
    }

    // The other two axes are bit-matrix transposes of the x columns, instead
    // of scattering every voxel's bit into three arrays.
    std::array<u32, 32> m;
    for (u32 y = 0; y < kChunkSize; ++y) {
        // Layer y: rows z, bits x  ->  rows x, bits z = z columns.
        std::copy_n(&cols.x[y * kChunkSize], kChunkSize, m.begin());
        transpose32(m);
        std::copy_n(m.begin(), kChunkSize, &cols.z[y * kChunkSize]);
    }
    for (u32 z = 0; z < kChunkSize; ++z) {
        // Slice z: rows y, bits x  ->  rows x, bits y = y columns.
        for (u32 y = 0; y < kChunkSize; ++y) {
            m[y] = cols.x[y * kChunkSize + z];
        }
        transpose32(m);
        std::copy_n(m.begin(), kChunkSize, &cols.y[z * kChunkSize]);
    }
}

[[nodiscard]] constexpr const std::array<u32, kChunkArea>& columnsFor(const Columns& cols, u32 axis) noexcept
{
    return axis == 0 ? cols.x : axis == 1 ? cols.y : cols.z;
}

// Visible-face masks of every column for one face direction, laid out like
// the columns ([a * 32 + b], bit = position along the normal).
using FaceMasks = std::array<u32, kChunkArea>;

// Fills `out` for face F and returns the OR of all masks: bit i set means
// slice i (along the normal) has at least one visible face. Zero = no faces.
template <Face F>
u32 computeFaceMasks(const Columns& cols, const BoundaryMasks& nb, FaceMasks& out) noexcept
{
    constexpr u32 axis = kFaceAxes[static_cast<usize>(F)].normal;
    constexpr bool positive = (static_cast<u32>(F) & 1u) == 0;
    const auto& columns = columnsFor(cols, axis);
    const auto& outsideRows = nb.rows[static_cast<usize>(F)];

#if defined(__AVX2__)
    // 8 columns per step. The neighbour bit of column b is bit b of the
    // boundary row, extracted with a per-lane variable shift.
    __m256i any = _mm256_setzero_si256();
    const __m256i one = _mm256_set1_epi32(1);
    for (u32 a = 0; a < kChunkSize; ++a) {
        const __m256i row = _mm256_set1_epi32(static_cast<i32>(outsideRows[a]));
        for (u32 b = 0; b < kChunkSize; b += 8) {
            const u32 idx = a * kChunkSize + b;
            const __m256i col = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&columns[idx]));
            const __m256i lane = _mm256_setr_epi32(
                static_cast<i32>(b), static_cast<i32>(b + 1), static_cast<i32>(b + 2), static_cast<i32>(b + 3),
                static_cast<i32>(b + 4), static_cast<i32>(b + 5), static_cast<i32>(b + 6), static_cast<i32>(b + 7));
            const __m256i outside = _mm256_and_si256(_mm256_srlv_epi32(row, lane), one);
            __m256i blocked;
            if constexpr (positive) {
                blocked = _mm256_or_si256(_mm256_srli_epi32(col, 1), _mm256_slli_epi32(outside, 31));
            } else {
                blocked = _mm256_or_si256(_mm256_slli_epi32(col, 1), outside);
            }
            const __m256i visible = _mm256_andnot_si256(blocked, col);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&out[idx]), visible);
            any = _mm256_or_si256(any, visible);
        }
    }
    // Horizontal OR of the 8 lanes.
    __m128i x = _mm_or_si128(_mm256_castsi256_si128(any), _mm256_extracti128_si256(any, 1));
    x = _mm_or_si128(x, _mm_shuffle_epi32(x, 0x4E));
    x = _mm_or_si128(x, _mm_shuffle_epi32(x, 0xB1));
    return static_cast<u32>(_mm_cvtsi128_si32(x));
#else
    u32 any = 0;
    for (u32 a = 0; a < kChunkSize; ++a) {
        const u32 row = outsideRows[a];
        for (u32 b = 0; b < kChunkSize; ++b) {
            const u32 col = columns[a * kChunkSize + b];
            const u32 outside = (row >> b) & 1u;
            const u32 visible = positive ? col & ~((col >> 1) | (outside << 31)) : col & ~((col << 1) | outside);
            out[a * kChunkSize + b] = visible;
            any |= visible;
        }
    }
    return any;
#endif
}

[[nodiscard]] u32 countFaces(const FaceMasks& masks) noexcept
{
    u32 count = 0;
    for (const u32 m : masks) {
        count += static_cast<u32>(std::popcount(m));
    }
    return count;
}

template <Face F>
void emitUnitQuads(const FaceMasks& masks, const BlockId* ENGINE_RESTRICT blocks, PackedQuad*& dst) noexcept
{
    constexpr u32 axis = kFaceAxes[static_cast<usize>(F)].normal;
    constexpr u32 axisA = kColumnAxes[axis][0];
    constexpr u32 axisB = kColumnAxes[axis][1];
    for (u32 a = 0; a < kChunkSize; ++a) {
        for (u32 b = 0; b < kChunkSize; ++b) {
            u32 mask = masks[a * kChunkSize + b];
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
                     FaceMasks& masks, std::vector<PackedQuad>& out)
{
    constexpr FaceAxes axes = kFaceAxes[static_cast<usize>(F)];
    constexpr u32 axisA = kColumnAxes[axes.normal][0];
    constexpr u32 axisB = kColumnAxes[axes.normal][1];
    constexpr u32 strideN = kAxisStride[axes.normal];
    constexpr u32 strideU = kAxisStride[axes.u];
    constexpr u32 strideV = kAxisStride[axes.v];

    const u32 slicesWithFaces = computeFaceMasks<F>(cols, nb, masks);
    if (slicesWithFaces == 0) {
        return; // e.g. the bottom of a chunk resting on solid ground
    }

    std::array<std::array<u32, kChunkSize>, kChunkSize> planes; // [slice][v], 4 KiB
    {
        ENGINE_PROFILE_SCOPE("scatterPlanes");
        for (u32 bits = slicesWithFaces; bits != 0; bits &= bits - 1) {
            planes[static_cast<usize>(std::countr_zero(bits))].fill(0); // only slices we will use
        }
        for (u32 a = 0; a < kChunkSize; ++a) {
            for (u32 b = 0; b < kChunkSize; ++b) {
                u32 mask = masks[a * kChunkSize + b];
                if (mask == 0) {
                    continue;
                }
                // (a, b) are the column's in-plane coordinates; map them to (u, v).
                std::array<u32, 3> c{};
                c[axisA] = a;
                c[axisB] = b;
                const u32 uBit = 1u << c[axes.u];
                const u32 v = c[axes.v];
                do {
                    const u32 slice = static_cast<u32>(std::countr_zero(mask));
                    mask &= mask - 1;
                    planes[slice][v] |= uBit;
                } while (mask != 0);
            }
        }
    }

    ENGINE_PROFILE_SCOPE("mergeQuads");
    for (u32 bits = slicesWithFaces; bits != 0; bits &= bits - 1) {
        const u32 slice = static_cast<u32>(std::countr_zero(bits));
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

OuterLayers computeOuterLayers(const Chunk& chunk)
{
    ENGINE_PROFILE_FUNCTION();
    OuterLayers out;
    auto& r = out.rows;
    const BlockId* blocks = chunk.blocks().data();
    constexpr u32 last = kChunkSize - 1;
    for (u32 y = 0; y < kChunkSize; ++y) {
        u32 xLow = 0;  // x = 0 layer, row y: bit z
        u32 xHigh = 0; // x = 31 layer
        for (u32 z = 0; z < kChunkSize; ++z) {
            const u32 m = solidMask32(blocks + voxelIndex(0, y, z)); // bit x
            xLow |= (m & 1u) << z;
            xHigh |= (m >> last) << z;
            if (y == 0) {
                r[static_cast<usize>(Face::NegY)][z] = m; // rows z, bits x
            } else if (y == last) {
                r[static_cast<usize>(Face::PosY)][z] = m;
            }
            if (z == 0) {
                r[static_cast<usize>(Face::NegZ)][y] = m; // rows y, bits x
            } else if (z == last) {
                r[static_cast<usize>(Face::PosZ)][y] = m;
            }
        }
        r[static_cast<usize>(Face::NegX)][y] = xLow; // rows y, bits z
        r[static_cast<usize>(Face::PosX)][y] = xHigh;
    }
    return out;
}

BoundaryMasks boundaryFromNeighbours(const std::array<const OuterLayers*, kFaceCount>& neighbours) noexcept
{
    BoundaryMasks masks;
    for (u32 f = 0; f < kFaceCount; ++f) {
        if (neighbours[f] != nullptr) {
            masks.rows[f] = neighbours[f]->rows[f ^ 1u];
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

    // Visible faces per direction, computed once (24 KiB).
    std::array<FaceMasks, kFaceCount> masks;
    computeFaceMasks<Face::PosX>(cols, neighbours, masks[0]);
    computeFaceMasks<Face::NegX>(cols, neighbours, masks[1]);
    computeFaceMasks<Face::PosY>(cols, neighbours, masks[2]);
    computeFaceMasks<Face::NegY>(cols, neighbours, masks[3]);
    computeFaceMasks<Face::PosZ>(cols, neighbours, masks[4]);
    computeFaceMasks<Face::NegZ>(cols, neighbours, masks[5]);

    // Count first so the output is sized exactly once (no per-quad capacity checks).
    u32 count = 0;
    for (const FaceMasks& m : masks) {
        count += countFaces(m);
    }
    out.resize(count);

    ENGINE_PROFILE_SCOPE("emitQuads");
    const BlockId* blocks = chunk.blocks().data();
    PackedQuad* dst = out.data();
    emitUnitQuads<Face::PosX>(masks[0], blocks, dst);
    emitUnitQuads<Face::NegX>(masks[1], blocks, dst);
    emitUnitQuads<Face::PosY>(masks[2], blocks, dst);
    emitUnitQuads<Face::NegY>(masks[3], blocks, dst);
    emitUnitQuads<Face::PosZ>(masks[4], blocks, dst);
    emitUnitQuads<Face::NegZ>(masks[5], blocks, dst);
    ENGINE_ASSERT(dst == out.data() + out.size());
}

void meshChunkGreedy(const Chunk& chunk, const BoundaryMasks& neighbours, std::vector<PackedQuad>& out)
{
    ENGINE_PROFILE_FUNCTION();
    out.clear();

    Columns cols;
    buildColumns(chunk, cols);

    const BlockId* blocks = chunk.blocks().data();
    FaceMasks masks; // reused for each direction
    emitGreedyQuads<Face::PosX>(cols, neighbours, blocks, masks, out);
    emitGreedyQuads<Face::NegX>(cols, neighbours, blocks, masks, out);
    emitGreedyQuads<Face::PosY>(cols, neighbours, blocks, masks, out);
    emitGreedyQuads<Face::NegY>(cols, neighbours, blocks, masks, out);
    emitGreedyQuads<Face::PosZ>(cols, neighbours, blocks, masks, out);
    emitGreedyQuads<Face::NegZ>(cols, neighbours, blocks, masks, out);
}

} // namespace engine::voxel
