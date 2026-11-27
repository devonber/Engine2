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
constexpr std::array<Offset, 6> kFaceOffsets{{
    {+1, 0, 0},
    {-1, 0, 0},
    {0, +1, 0},
    {0, -1, 0},
    {0, 0, +1},
    {0, 0, -1},
}};

// Occupancy bitmasks: one u32 per column of 32 voxels along an axis.
//   x[y*32 + z] bit i -> voxel (i, y, z)
//   y[z*32 + x] bit i -> voxel (x, i, z)
//   z[y*32 + x] bit i -> voxel (x, y, i)
struct Columns {
    std::array<u32, kChunkArea> x;
    std::array<u32, kChunkArea> y;
    std::array<u32, kChunkArea> z;
};

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

// A face is visible where the column is solid and its neighbour (the same
// column shifted by one) is not. Shifting in zeros treats the border as air.
[[nodiscard]] constexpr u32 posFaces(u32 col) noexcept
{
    return col & ~(col >> 1);
}
[[nodiscard]] constexpr u32 negFaces(u32 col) noexcept
{
    return col & ~(col << 1);
}

} // namespace

void meshChunkNaive(const Chunk& chunk, std::vector<PackedQuad>& out)
{
    ENGINE_PROFILE_FUNCTION();
    out.clear();

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
                    const bool neighbourSolid =
                        inChunk(nx, ny, nz) &&
                        chunk.get(static_cast<u32>(nx), static_cast<u32>(ny), static_cast<u32>(nz)) != kAir;
                    if (!neighbourSolid) {
                        out.push_back(PackedQuad::make(x, y, z, static_cast<Face>(f), block));
                    }
                }
            }
        }
    }
}

void meshChunkBinary(const Chunk& chunk, std::vector<PackedQuad>& out)
{
    ENGINE_PROFILE_FUNCTION();

    Columns cols; // 12 KiB, intentionally uninitialised (buildColumns writes it)
    buildColumns(chunk, cols);

    // Pass 1: count faces so the output is sized exactly once. Popcount is a
    // single instruction, so this costs far less than per-face push_back checks.
    usize count = 0;
    for (u32 i = 0; i < kChunkArea; ++i) {
        count += static_cast<usize>(std::popcount(posFaces(cols.x[i])) + std::popcount(negFaces(cols.x[i])) +
                                    std::popcount(posFaces(cols.y[i])) + std::popcount(negFaces(cols.y[i])) +
                                    std::popcount(posFaces(cols.z[i])) + std::popcount(negFaces(cols.z[i])));
    }
    out.resize(count);

    // Pass 2: walk set bits and write quads straight into the buffer.
    {
        ENGINE_PROFILE_SCOPE("emitQuads");
        const BlockId* ENGINE_RESTRICT blocks = chunk.blocks().data();
        PackedQuad* ENGINE_RESTRICT dst = out.data();

        const auto emit = [&](u32 mask, Face face, auto toXYZ) {
            while (mask != 0) {
                const u32 i = static_cast<u32>(std::countr_zero(mask));
                mask &= mask - 1; // clear lowest set bit
                const auto [x, y, z] = toXYZ(i);
                *dst++ = PackedQuad::make(x, y, z, face, blocks[voxelIndex(x, y, z)]);
            }
        };

        struct XYZ {
            u32 x, y, z;
        };

        for (u32 a = 0; a < kChunkSize; ++a) {
            for (u32 b = 0; b < kChunkSize; ++b) {
                const u32 idx = a * kChunkSize + b;

                const u32 cx = cols.x[idx]; // a = y, b = z
                const auto alongX = [a, b](u32 i) { return XYZ{i, a, b}; };
                emit(posFaces(cx), Face::PosX, alongX);
                emit(negFaces(cx), Face::NegX, alongX);

                const u32 cy = cols.y[idx]; // a = z, b = x
                const auto alongY = [a, b](u32 i) { return XYZ{b, i, a}; };
                emit(posFaces(cy), Face::PosY, alongY);
                emit(negFaces(cy), Face::NegY, alongY);

                const u32 cz = cols.z[idx]; // a = y, b = x
                const auto alongZ = [a, b](u32 i) { return XYZ{b, a, i}; };
                emit(posFaces(cz), Face::PosZ, alongZ);
                emit(negFaces(cz), Face::NegZ, alongZ);
            }
        }

        ENGINE_ASSERT(dst == out.data() + out.size());
    }
}

} // namespace engine::voxel
