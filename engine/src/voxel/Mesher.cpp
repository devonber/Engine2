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

// Solidity of the chunk plus its one-voxel shell (a 34^3 bit volume), for
// ambient occlusion. Stored twice, as rows of bits along x and along z, so
// that for every face direction the voxels around a face lie in 3 adjacent
// rows of one of them:
//   x[(y + 1) * 34 + (z + 1)] bit (x + 1)   -> faces facing +-Y and +-Z
//   z[(y + 1) * 34 + (x + 1)] bit (z + 1)   -> faces facing +-X
// for x, y, z in -1..32.
struct Padded {
    static constexpr i32 kSide = static_cast<i32>(kChunkSize) + 2;
    std::array<u64, kSide * kSide> x;
    std::array<u64, kSide * kSide> z;

    [[nodiscard]] static constexpr usize index(i32 row, i32 col) noexcept
    {
        return static_cast<usize>((row + 1) * kSide + (col + 1));
    }
};

[[nodiscard]] constexpr usize edgeIndex(bool a1High, bool a2High) noexcept
{
    return (a1High ? 1u : 0u) | (a2High ? 2u : 0u);
}

void buildPadded(const Columns& cols, const NeighbourShell& shell, Padded& out) noexcept
{
    ENGINE_PROFILE_SCOPE("buildPadded");
    const auto& f = shell.faces.rows;
    constexpr i32 n = static_cast<i32>(kChunkSize);
    const auto pack = [](u32 inner, u32 low, u32 high) {
        return (static_cast<u64>(inner) << 1) | low | (static_cast<u64>(high) << 33);
    };

    // Rows along x, indexed (y, z).
    for (i32 y = -1; y <= n; ++y) {
        const bool yIn = y >= 0 && y < n;
        for (i32 z = -1; z <= n; ++z) {
            const bool zIn = z >= 0 && z < n;
            u32 inner = 0; // x = 0..31
            u32 low = 0;   // x = -1
            u32 high = 0;  // x = 32
            if (yIn && zIn) {
                inner = cols.x[static_cast<usize>(y) * kChunkSize + static_cast<usize>(z)];
                low = (f[static_cast<usize>(Face::NegX)][static_cast<usize>(y)] >> z) & 1u; // rows y, bits z
                high = (f[static_cast<usize>(Face::PosX)][static_cast<usize>(y)] >> z) & 1u;
            } else if (zIn) { // y outside: the -Y / +Y face layer (rows z, bits x)
                inner = f[static_cast<usize>(y < 0 ? Face::NegY : Face::PosY)][static_cast<usize>(z)];
                // ends: edges along z (a1 = x, a2 = y)
                low = (shell.edges[2][edgeIndex(false, y >= n)] >> z) & 1u;
                high = (shell.edges[2][edgeIndex(true, y >= n)] >> z) & 1u;
            } else if (yIn) { // z outside: the -Z / +Z face layer (rows y, bits x)
                inner = f[static_cast<usize>(z < 0 ? Face::NegZ : Face::PosZ)][static_cast<usize>(y)];
                // ends: edges along y (a1 = z, a2 = x)
                low = (shell.edges[1][edgeIndex(z >= n, false)] >> y) & 1u;
                high = (shell.edges[1][edgeIndex(z >= n, true)] >> y) & 1u;
            } else { // y and z outside: the edge along x (a1 = y, a2 = z), corners at the ends
                inner = shell.edges[0][edgeIndex(y >= n, z >= n)];
                const u32 yz = (y >= n ? 2u : 0u) | (z >= n ? 4u : 0u);
                low = (shell.corners >> yz) & 1u;
                high = (shell.corners >> (yz | 1u)) & 1u;
            }
            out.x[Padded::index(y, z)] = pack(inner, low, high);
        }
    }

    // Rows along z, indexed (y, x).
    for (i32 y = -1; y <= n; ++y) {
        const bool yIn = y >= 0 && y < n;
        for (i32 x = -1; x <= n; ++x) {
            const bool xIn = x >= 0 && x < n;
            u32 inner = 0; // z = 0..31
            u32 low = 0;   // z = -1
            u32 high = 0;  // z = 32
            if (yIn && xIn) {
                inner = cols.z[static_cast<usize>(y) * kChunkSize + static_cast<usize>(x)];
                low = (f[static_cast<usize>(Face::NegZ)][static_cast<usize>(y)] >> x) & 1u; // rows y, bits x
                high = (f[static_cast<usize>(Face::PosZ)][static_cast<usize>(y)] >> x) & 1u;
            } else if (yIn) { // x outside: the -X / +X face layer (rows y, bits z)
                inner = f[static_cast<usize>(x < 0 ? Face::NegX : Face::PosX)][static_cast<usize>(y)];
                // ends: edges along y (a1 = z, a2 = x)
                low = (shell.edges[1][edgeIndex(false, x >= n)] >> y) & 1u;
                high = (shell.edges[1][edgeIndex(true, x >= n)] >> y) & 1u;
            } else if (xIn) { // y outside: the -Y / +Y face layer (rows z, bits x): gather bit x
                const auto& layer = f[static_cast<usize>(y < 0 ? Face::NegY : Face::PosY)];
                for (u32 zz = 0; zz < kChunkSize; ++zz) {
                    inner |= ((layer[zz] >> x) & 1u) << zz;
                }
                // ends: edges along x (a1 = y, a2 = z)
                low = (shell.edges[0][edgeIndex(y >= n, false)] >> x) & 1u;
                high = (shell.edges[0][edgeIndex(y >= n, true)] >> x) & 1u;
            } else { // y and x outside: the edge along z (a1 = x, a2 = y), corners at the ends
                inner = shell.edges[2][edgeIndex(x >= n, y >= n)];
                const u32 xy = (x >= n ? 1u : 0u) | (y >= n ? 2u : 0u);
                low = (shell.corners >> xy) & 1u;
                high = (shell.corners >> (xy | 4u)) & 1u;
            }
            out.z[Padded::index(y, x)] = pack(inner, low, high);
        }
    }
}

// Occlusion of one corner from the three voxels touching it in front of the
// face: two sides and the diagonal. Both sides solid hides the diagonal
// (it cannot be seen), so that is fully occluded.
[[nodiscard]] constexpr u32 cornerAO(bool side1, bool side2, bool corner) noexcept
{
    return (side1 && side2) ? 0u : 3u - (static_cast<u32>(side1) + static_cast<u32>(side2) + static_cast<u32>(corner));
}

// All four corner levels of a face from the 3x3 voxels in front of it, as a
// 512-entry table indexed by the 9 occupancy bits (bit (l + 1) + 3 * (r + 1)
// for lane offset l and row offset r in -1..1). `laneIsU`: whether the lane
// (bit) direction is the face's U axis or its V axis.
template <bool LaneIsU>
constexpr std::array<u8, 512> makeAOTable() noexcept
{
    std::array<u8, 512> table{};
    for (u32 pattern = 0; pattern < 512; ++pattern) {
        // Solid at offset (du, dv) along the face's U and V axes.
        const auto solid = [&](i32 du, i32 dv) {
            const i32 l = LaneIsU ? du : dv;
            const i32 r = LaneIsU ? dv : du;
            return ((pattern >> static_cast<u32>((l + 1) + 3 * (r + 1))) & 1u) != 0;
        };
        u32 ao = 0;
        for (u32 corner = 0; corner < 4; ++corner) {
            const i32 du = (corner & 1u) != 0 ? 1 : -1;
            const i32 dv = (corner & 2u) != 0 ? 1 : -1;
            ao |= cornerAO(solid(du, 0), solid(0, dv), solid(du, dv)) << (2 * corner);
        }
        table[pattern] = static_cast<u8>(ao);
    }
    return table;
}
constexpr std::array<u8, 512> kAOLaneU = makeAOTable<true>();
constexpr std::array<u8, 512> kAOLaneV = makeAOTable<false>();

// The four corner levels of face F of voxel (x, y, z), packed as in
// PackedQuad: three padded rows, a 3-bit window from each, one table lookup.
template <Face F>
[[nodiscard]] ENGINE_FORCEINLINE u32 faceAO(const Padded& p, i32 x, i32 y, i32 z) noexcept
{
    constexpr FaceAxes axes = kFaceAxes[static_cast<usize>(F)];
    constexpr i32 s = (static_cast<u32>(F) & 1u) == 0 ? 1 : -1;
    // Lanes run along x for Y and Z faces, along z for X faces; rows are the
    // remaining in-plane axis.
    constexpr u32 laneAxis = axes.normal == 0 ? 2u : 0u;
    constexpr bool laneIsU = axes.u == laneAxis;

    u64 below = 0, mid = 0, above = 0;
    i32 lane = 0;
    if constexpr (axes.normal == 1) { // Y faces: layer y + s, rows z, bits x
        const i32 fy = y + s;
        below = p.x[Padded::index(fy, z - 1)];
        mid = p.x[Padded::index(fy, z)];
        above = p.x[Padded::index(fy, z + 1)];
        lane = x;
    } else if constexpr (axes.normal == 2) { // Z faces: layer z + s, rows y, bits x
        const i32 fz = z + s;
        below = p.x[Padded::index(y - 1, fz)];
        mid = p.x[Padded::index(y, fz)];
        above = p.x[Padded::index(y + 1, fz)];
        lane = x;
    } else { // X faces: layer x + s, rows y, bits z
        const i32 fx = x + s;
        below = p.z[Padded::index(y - 1, fx)];
        mid = p.z[Padded::index(y, fx)];
        above = p.z[Padded::index(y + 1, fx)];
        lane = z;
    }
    // Bits lane, lane + 1, lane + 2 of a padded row = lane offsets -1, 0, +1.
    const u32 l = static_cast<u32>(lane);
    const u32 pattern = static_cast<u32>((below >> l) & 7u) | static_cast<u32>(((mid >> l) & 7u) << 3) |
                        static_cast<u32>(((above >> l) & 7u) << 6);
    return laneIsU ? kAOLaneU[pattern] : kAOLaneV[pattern];
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
void emitUnitQuads(const FaceMasks& masks, const BlockId* ENGINE_RESTRICT blocks, const Padded* padded,
                   PackedQuad*& dst) noexcept
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
                const u32 ao = padded != nullptr ? faceAO<F>(*padded, static_cast<i32>(c[0]), static_cast<i32>(c[1]),
                                                             static_cast<i32>(c[2]))
                                                 : PackedQuad::kNoOcclusion;
                *dst++ = PackedQuad::make(c[0], c[1], c[2], F, blocks[voxelIndex(c[0], c[1], c[2])], 1, 1, ao);
            }
        }
    }
}

// Greedy meshing for one face direction.
//
// 1. Scatter every visible face into a 2D bit-plane per slice along the
//    normal: plane[slice][v] bit u. Each plane row is 32 faces.
// 2. Per row, take the lowest set bit, grow it along U while the bits are set
//    and the block (and occlusion) matches, then grow the run along V while
//    the next row has the whole run set with the same block and occlusion.
//    Clear it and emit one quad.
//
// Equal occlusion is what keeps merged quads shaded exactly like separate
// faces: neighbouring faces share their corner vertices, so equal levels
// along the merge direction mean the shading is constant along it.
template <Face F>
void emitGreedyQuads(const Columns& cols, const BoundaryMasks& nb, const BlockId* ENGINE_RESTRICT blocks,
                     const Padded* padded, FaceMasks& masks, std::vector<PackedQuad>& out)
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
        // Occlusion of every face in this slice, computed once per face up
        // front (merging would otherwise recompute it for each face it tries).
        std::array<std::array<u8, kChunkSize>, kChunkSize> aoPlane; // [v][u]; only visible faces are written
        if (padded != nullptr) {
            std::array<i32, 3> c{};
            c[axes.normal] = static_cast<i32>(slice);
            for (u32 v = 0; v < kChunkSize; ++v) {
                c[axes.v] = static_cast<i32>(v);
                for (u32 bitsLeft = rows[v]; bitsLeft != 0; bitsLeft &= bitsLeft - 1) {
                    const u32 u = static_cast<u32>(std::countr_zero(bitsLeft));
                    c[axes.u] = static_cast<i32>(u);
                    aoPlane[v][u] = static_cast<u8>(faceAO<F>(*padded, c[0], c[1], c[2]));
                }
            }
        }
        const auto aoAt = [&](u32 u, u32 v) -> u32 {
            return padded != nullptr ? aoPlane[v][u] : PackedQuad::kNoOcclusion;
        };
        for (u32 v = 0; v < kChunkSize; ++v) {
            while (rows[v] != 0) {
                const u32 row = rows[v];
                const u32 u0 = static_cast<u32>(std::countr_zero(row));
                const u32 base = slice * strideN + u0 * strideU + v * strideV;
                const BlockId block = blocks[base];
                const u32 ao = aoAt(u0, v);

                // Grow along U: contiguous set bits with the same block and occlusion.
                const u32 runLength = static_cast<u32>(std::countr_one(row >> u0));
                u32 width = 1;
                while (width < runLength && blocks[base + width * strideU] == block && aoAt(u0 + width, v) == ao) {
                    ++width;
                }
                const u32 runMask = (width == 32 ? ~0u : ((1u << width) - 1u)) << u0;

                // Grow along V: next row must contain the whole run, all the same block.
                u32 height = 1;
                while (v + height < kChunkSize && (rows[v + height] & runMask) == runMask) {
                    const u32 rowBase = base + height * strideV;
                    bool same = true;
                    for (u32 k = 0; k < width && same; ++k) {
                        same = blocks[rowBase + k * strideU] == block && aoAt(u0 + k, v + height) == ao;
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
                out.push_back(PackedQuad::make(c[0], c[1], c[2], F, block, width, height, ao));
            }
        }
    }
}

} // namespace

void NeighbourShell::setAllSolid() noexcept
{
    for (auto& face : faces.rows) {
        face.fill(~0u);
    }
    for (auto& axis : edges) {
        axis.fill(~0u);
    }
    corners = 0xFF;
}

namespace {

// Fills the 12 edges and 8 corners of a shell from a callback telling whether
// neighbour (dx, dy, dz) is solid at its local voxel (lx, ly, lz).
template <typename SolidAt>
void fillEdgesAndCorners(NeighbourShell& shell, SolidAt&& solidAt)
{
    constexpr u32 last = kChunkSize - 1;
    for (u32 c = 0; c < 3; ++c) {
        const u32 a1 = (c + 1) % 3;
        const u32 a2 = (c + 2) % 3;
        for (u32 i = 0; i < 4; ++i) {
            std::array<i32, 3> d{};
            std::array<u32, 3> local{};
            d[a1] = (i & 1u) != 0 ? 1 : -1;
            d[a2] = (i & 2u) != 0 ? 1 : -1;
            local[a1] = d[a1] > 0 ? 0 : last; // the neighbour's voxel touching us
            local[a2] = d[a2] > 0 ? 0 : last;
            u32 bits = 0;
            for (u32 k = 0; k < kChunkSize; ++k) {
                local[c] = k;
                bits |= (solidAt(d[0], d[1], d[2], local[0], local[1], local[2]) ? 1u : 0u) << k;
            }
            shell.edges[c][i] = bits;
        }
    }
    shell.corners = 0;
    for (u32 i = 0; i < 8; ++i) {
        const i32 dx = (i & 1u) != 0 ? 1 : -1;
        const i32 dy = (i & 2u) != 0 ? 1 : -1;
        const i32 dz = (i & 4u) != 0 ? 1 : -1;
        const bool solid = solidAt(dx, dy, dz, dx > 0 ? 0 : last, dy > 0 ? 0 : last, dz > 0 ? 0 : last);
        shell.corners |= (solid ? 1u : 0u) << i;
    }
}

// Whether a voxel on a chunk's outer shell is solid, from its OuterLayers.
[[nodiscard]] bool outerSolid(const OuterLayers& layers, u32 x, u32 y, u32 z) noexcept
{
    constexpr u32 last = kChunkSize - 1;
    const auto& r = layers.rows;
    if (x == 0 || x == last) {
        return ((r[static_cast<usize>(x == 0 ? Face::NegX : Face::PosX)][y] >> z) & 1u) != 0; // rows y, bits z
    }
    if (y == 0 || y == last) {
        return ((r[static_cast<usize>(y == 0 ? Face::NegY : Face::PosY)][z] >> x) & 1u) != 0; // rows z, bits x
    }
    return ((r[static_cast<usize>(z == 0 ? Face::NegZ : Face::PosZ)][y] >> x) & 1u) != 0; // rows y, bits x
}

[[nodiscard]] constexpr std::array<i32, 3> faceOffset(u32 f) noexcept
{
    std::array<i32, 3> d{};
    d[kFaceAxes[f].normal] = (f & 1u) == 0 ? 1 : -1;
    return d;
}

} // namespace

NeighbourShell buildNeighbourShell(const ChunkNeighbourhood& neighbours)
{
    ENGINE_PROFILE_FUNCTION();
    NeighbourShell shell;
    const auto solidAt = [&](i32 dx, i32 dy, i32 dz, u32 x, u32 y, u32 z) {
        const Chunk* n = neighbours.chunks[neighbourIndex(dx, dy, dz)];
        return n != nullptr && n->get(x, y, z) != kAir;
    };
    // Faces: the neighbour's layer touching this chunk, voxel by voxel.
    constexpr u32 last = kChunkSize - 1;
    for (u32 f = 0; f < kFaceCount; ++f) {
        const auto d = faceOffset(f);
        const u32 normal = kFaceAxes[f].normal;
        const u32 rowAxis = kColumnAxes[normal][0];
        const u32 colAxis = kColumnAxes[normal][1];
        for (u32 row = 0; row < kChunkSize; ++row) {
            u32 bits = 0;
            for (u32 col = 0; col < kChunkSize; ++col) {
                std::array<u32, 3> local{};
                local[normal] = d[normal] > 0 ? 0 : last;
                local[rowAxis] = row;
                local[colAxis] = col;
                bits |= (solidAt(d[0], d[1], d[2], local[0], local[1], local[2]) ? 1u : 0u) << col;
            }
            shell.faces.rows[f][row] = bits;
        }
    }
    fillEdgesAndCorners(shell, solidAt);
    return shell;
}

NeighbourShell shellFromOuterLayers(const std::array<const OuterLayers*, kNeighbourhoodSize>& neighbours) noexcept
{
    NeighbourShell shell;
    // Faces: the neighbour on side f touches with its opposite layer (f ^ 1).
    for (u32 f = 0; f < kFaceCount; ++f) {
        const auto d = faceOffset(f);
        if (const OuterLayers* n = neighbours[neighbourIndex(d[0], d[1], d[2])]) {
            shell.faces.rows[f] = n->rows[f ^ 1u];
        }
    }
    fillEdgesAndCorners(shell, [&](i32 dx, i32 dy, i32 dz, u32 x, u32 y, u32 z) {
        const OuterLayers* n = neighbours[neighbourIndex(dx, dy, dz)];
        return n != nullptr && outerSolid(*n, x, y, z);
    });
    return shell;
}

const OuterLayers& solidOuterLayers() noexcept
{
    static const OuterLayers solid = [] {
        OuterLayers o;
        for (auto& face : o.rows) {
            face.fill(~0u);
        }
        return o;
    }();
    return solid;
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

void meshChunkNaive(const Chunk& chunk, const ChunkNeighbourhood& neighbours, std::vector<PackedQuad>& out,
                    bool ambientOcclusion)
{
    ENGINE_PROFILE_FUNCTION();
    out.clear();
    constexpr i32 n = static_cast<i32>(kChunkSize);

    // Solid at chunk-local (x, y, z), which may be up to one chunk outside.
    const auto solid = [&](i32 x, i32 y, i32 z) {
        const auto chunkOf = [](i32 v) { return v < 0 ? -1 : (v >= n ? 1 : 0); };
        const i32 cx = chunkOf(x), cy = chunkOf(y), cz = chunkOf(z);
        const Chunk* src = (cx | cy | cz) == 0 ? &chunk : neighbours.chunks[neighbourIndex(cx, cy, cz)];
        return src != nullptr && src->get(static_cast<u32>(x - cx * n), static_cast<u32>(y - cy * n),
                                          static_cast<u32>(z - cz * n)) != kAir;
    };

    for (i32 y = 0; y < n; ++y) {
        for (i32 z = 0; z < n; ++z) {
            for (i32 x = 0; x < n; ++x) {
                const BlockId block = chunk.get(static_cast<u32>(x), static_cast<u32>(y), static_cast<u32>(z));
                if (block == kAir) {
                    continue;
                }
                for (u32 f = 0; f < kFaceCount; ++f) {
                    const FaceAxes axes = kFaceAxes[f];
                    std::array<i32, 3> front{x, y, z};
                    front[axes.normal] += (f & 1u) == 0 ? 1 : -1;
                    if (solid(front[0], front[1], front[2])) {
                        continue;
                    }
                    u32 ao = PackedQuad::kNoOcclusion;
                    if (ambientOcclusion) {
                        // By definition, straight from the voxels.
                        ao = 0;
                        for (u32 corner = 0; corner < 4; ++corner) {
                            const i32 du = (corner & 1u) != 0 ? 1 : -1;
                            const i32 dv = (corner & 2u) != 0 ? 1 : -1;
                            const auto at = [&](i32 su, i32 sv) {
                                std::array<i32, 3> c = front;
                                c[axes.u] += su;
                                c[axes.v] += sv;
                                return solid(c[0], c[1], c[2]);
                            };
                            const bool side1 = at(du, 0), side2 = at(0, dv), diag = at(du, dv);
                            const u32 level = (side1 && side2) ? 0u : 3u - (side1 + side2 + diag);
                            ao |= level << (2 * corner);
                        }
                    }
                    out.push_back(PackedQuad::make(static_cast<u32>(x), static_cast<u32>(y), static_cast<u32>(z),
                                                   static_cast<Face>(f), block, 1, 1, ao));
                }
            }
        }
    }
}

void meshChunkBinary(const Chunk& chunk, const NeighbourShell& shell, std::vector<PackedQuad>& out,
                     bool ambientOcclusion)
{
    ENGINE_PROFILE_FUNCTION();

    Columns cols; // 12 KiB, intentionally uninitialised (buildColumns writes it)
    buildColumns(chunk, cols);

    // Visible faces per direction, computed once (24 KiB).
    std::array<FaceMasks, kFaceCount> masks;
    const BoundaryMasks& nb = shell.faces;
    computeFaceMasks<Face::PosX>(cols, nb, masks[0]);
    computeFaceMasks<Face::NegX>(cols, nb, masks[1]);
    computeFaceMasks<Face::PosY>(cols, nb, masks[2]);
    computeFaceMasks<Face::NegY>(cols, nb, masks[3]);
    computeFaceMasks<Face::PosZ>(cols, nb, masks[4]);
    computeFaceMasks<Face::NegZ>(cols, nb, masks[5]);

    // Count first so the output is sized exactly once (no per-quad capacity checks).
    u32 count = 0;
    for (const FaceMasks& m : masks) {
        count += countFaces(m);
    }
    out.resize(count);

    Padded padded;
    if (ambientOcclusion && count > 0) {
        buildPadded(cols, shell, padded);
    }
    const Padded* pad = ambientOcclusion ? &padded : nullptr;

    ENGINE_PROFILE_SCOPE("emitQuads");
    const BlockId* blocks = chunk.blocks().data();
    PackedQuad* dst = out.data();
    emitUnitQuads<Face::PosX>(masks[0], blocks, pad, dst);
    emitUnitQuads<Face::NegX>(masks[1], blocks, pad, dst);
    emitUnitQuads<Face::PosY>(masks[2], blocks, pad, dst);
    emitUnitQuads<Face::NegY>(masks[3], blocks, pad, dst);
    emitUnitQuads<Face::PosZ>(masks[4], blocks, pad, dst);
    emitUnitQuads<Face::NegZ>(masks[5], blocks, pad, dst);
    ENGINE_ASSERT(dst == out.data() + out.size());
}

void meshChunkGreedy(const Chunk& chunk, const NeighbourShell& shell, std::vector<PackedQuad>& out,
                     bool ambientOcclusion)
{
    ENGINE_PROFILE_FUNCTION();
    out.clear();

    Columns cols;
    buildColumns(chunk, cols);

    Padded padded; // ~9 KiB, built only when needed
    if (ambientOcclusion) {
        buildPadded(cols, shell, padded);
    }
    const Padded* pad = ambientOcclusion ? &padded : nullptr;

    const BlockId* blocks = chunk.blocks().data();
    const BoundaryMasks& nb = shell.faces;
    FaceMasks masks; // reused for each direction
    emitGreedyQuads<Face::PosX>(cols, nb, blocks, pad, masks, out);
    emitGreedyQuads<Face::NegX>(cols, nb, blocks, pad, masks, out);
    emitGreedyQuads<Face::PosY>(cols, nb, blocks, pad, masks, out);
    emitGreedyQuads<Face::NegY>(cols, nb, blocks, pad, masks, out);
    emitGreedyQuads<Face::PosZ>(cols, nb, blocks, pad, masks, out);
    emitGreedyQuads<Face::NegZ>(cols, nb, blocks, pad, masks, out);
}

} // namespace engine::voxel
