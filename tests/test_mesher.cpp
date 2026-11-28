#include "engine/core/Random.hpp"
#include "engine/voxel/Generate.hpp"
#include "engine/voxel/Mesher.hpp"

#include <doctest/doctest.h>

#include <algorithm>
#include <memory>
#include <vector>

using namespace engine;
using namespace engine::voxel;

namespace {

std::vector<PackedQuad> sorted(std::vector<PackedQuad> v)
{
    std::ranges::sort(v);
    return v;
}

// Split greedy rectangles back into one unit quad per block face.
std::vector<PackedQuad> expand(const std::vector<PackedQuad>& quads)
{
    std::vector<PackedQuad> unit;
    for (const PackedQuad& q : quads) {
        const FaceAxes axes = kFaceAxes[static_cast<usize>(q.face())];
        for (u32 j = 0; j < q.height(); ++j) {
            for (u32 i = 0; i < q.width(); ++i) {
                u32 c[3] = {q.x(), q.y(), q.z()};
                c[axes.u] += i;
                c[axes.v] += j;
                unit.push_back(PackedQuad::make(c[0], c[1], c[2], q.face(), q.block()));
            }
        }
    }
    return unit;
}

// Naive (reads neighbour chunks) is the reference for binary and greedy
// (which read boundary masks built from the same neighbours).
void checkMeshersAgree(const Chunk& chunk, const ChunkNeighbours& neighbours = {})
{
    const BoundaryMasks masks = buildBoundaryMasks(neighbours);
    std::vector<PackedQuad> naive, binary, greedy;
    meshChunkNaive(chunk, neighbours, naive);
    meshChunkBinary(chunk, masks, binary);
    meshChunkGreedy(chunk, masks, greedy);

    const auto reference = sorted(naive);
    REQUIRE(binary.size() == naive.size());
    CHECK(sorted(binary) == reference);

    // Same faces covered exactly once: no gaps, no overlaps.
    const auto greedyFaces = expand(greedy);
    REQUIRE(greedyFaces.size() == naive.size());
    CHECK(sorted(greedyFaces) == reference);
    CHECK(greedy.size() <= naive.size());
}

std::unique_ptr<Chunk> randomChunk(f64 density, u64 seed)
{
    auto c = std::make_unique<Chunk>();
    fillRandom(*c, density, seed);
    return c;
}

} // namespace

TEST_CASE("PackedQuad round-trips its fields")
{
    constexpr auto q = PackedQuad::make(31, 17, 3, Face::NegZ, 0xBEEF);
    static_assert(q.x() == 31 && q.y() == 17 && q.z() == 3);
    static_assert(q.face() == Face::NegZ && q.block() == 0xBEEF);
    static_assert(q.width() == 1 && q.height() == 1);

    constexpr auto big = PackedQuad::make(0, 5, 0, Face::PosY, 7, 32, 32);
    static_assert(big.width() == 32 && big.height() == 32 && big.y() == 5 && big.block() == 7);
}

TEST_CASE("Face axes are right-handed: U x V is the outward normal")
{
    // With axes in cyclic order (x, y, z), U x V = +N when V follows U, and
    // -N when U follows V. Positive faces need the first, negative the second.
    for (u32 f = 0; f < kFaceCount; ++f) {
        CAPTURE(f);
        const FaceAxes a = kFaceAxes[f];
        CHECK(a.normal == f / 2);
        CHECK(a.u != a.normal);
        CHECK(a.v != a.normal);
        if (f % 2 == 0) {
            CHECK((a.u + 1) % 3 == a.v);
        } else {
            CHECK((a.v + 1) % 3 == a.u);
        }
    }
}

TEST_CASE("Meshing simple shapes")
{
    auto chunk = std::make_unique<Chunk>();
    std::vector<PackedQuad> quads;
    const BoundaryMasks air{};

    SUBCASE("empty chunk has no faces")
    {
        meshChunkBinary(*chunk, air, quads);
        CHECK(quads.empty());
        meshChunkGreedy(*chunk, air, quads);
        CHECK(quads.empty());
    }
    SUBCASE("single block has 6 faces")
    {
        chunk->set(10, 10, 10, 1);
        meshChunkBinary(*chunk, air, quads);
        CHECK(quads.size() == 6);
        meshChunkGreedy(*chunk, air, quads);
        CHECK(quads.size() == 6);
    }
    SUBCASE("two adjacent blocks hide the shared faces")
    {
        chunk->set(10, 10, 10, 1);
        chunk->set(11, 10, 10, 1);
        meshChunkBinary(*chunk, air, quads);
        CHECK(quads.size() == 10);
        meshChunkGreedy(*chunk, air, quads);
        CHECK(quads.size() == 6); // the four long sides merge into 1x2 quads
    }
    SUBCASE("full chunk: one quad per side when greedy")
    {
        chunk->fill(1);
        meshChunkBinary(*chunk, air, quads);
        CHECK(quads.size() == 6 * kChunkArea);
        meshChunkGreedy(*chunk, air, quads);
        REQUIRE(quads.size() == 6);
        for (const auto& q : quads) {
            CHECK(q.width() == 32);
            CHECK(q.height() == 32);
        }
    }
    SUBCASE("different blocks do not merge")
    {
        for (u32 x = 0; x < kChunkSize; ++x) {
            chunk->set(x, 0, 0, x % 2 == 0 ? 1 : 2);
        }
        meshChunkGreedy(*chunk, air, quads);
        // Each block keeps its own top, bottom and two side faces; only the ends are shared.
        CHECK(quads.size() == 32 * 4 + 2);
    }
    SUBCASE("solid neighbours hide border faces")
    {
        chunk->fill(1);
        BoundaryMasks solid;
        for (u32 f = 0; f < kFaceCount; ++f) {
            solid.setSolid(static_cast<Face>(f));
        }
        meshChunkBinary(*chunk, solid, quads);
        CHECK(quads.empty());
        meshChunkGreedy(*chunk, solid, quads);
        CHECK(quads.empty());
    }
}

TEST_CASE("Fast meshers match the naive reference")
{
    auto chunk = std::make_unique<Chunk>();

    SUBCASE("terrain, no neighbours")
    {
        for (i32 c = -3; c <= 3; ++c) {
            fillTerrain(*chunk, c, c * 7);
            checkMeshersAgree(*chunk);
        }
    }
    SUBCASE("random densities, no neighbours")
    {
        for (const f64 density : {0.01, 0.1, 0.5, 0.9, 0.99}) {
            CAPTURE(density);
            fillRandom(*chunk, density, 1234);
            checkMeshersAgree(*chunk);
        }
    }
    SUBCASE("terrain with terrain neighbours")
    {
        auto px = std::make_unique<Chunk>(), nx = std::make_unique<Chunk>();
        auto pz = std::make_unique<Chunk>(), nz = std::make_unique<Chunk>();
        fillTerrain(*chunk, 2, 5);
        fillTerrain(*px, 3, 5);
        fillTerrain(*nx, 1, 5);
        fillTerrain(*pz, 2, 6);
        fillTerrain(*nz, 2, 4);
        ChunkNeighbours n;
        n.chunks[static_cast<usize>(Face::PosX)] = px.get();
        n.chunks[static_cast<usize>(Face::NegX)] = nx.get();
        n.chunks[static_cast<usize>(Face::PosZ)] = pz.get();
        n.chunks[static_cast<usize>(Face::NegZ)] = nz.get();
        checkMeshersAgree(*chunk, n);
    }
    SUBCASE("random chunk with random neighbours, some missing")
    {
        SplitMix64 rng{99};
        for (int trial = 0; trial < 8; ++trial) {
            CAPTURE(trial);
            fillRandom(*chunk, 0.2 + 0.6 * rng.nextUnit(), rng.next());
            std::vector<std::unique_ptr<Chunk>> owned;
            ChunkNeighbours n;
            for (u32 f = 0; f < kFaceCount; ++f) {
                if (rng.nextUnit() < 0.75) {
                    owned.push_back(randomChunk(rng.nextUnit(), rng.next()));
                    n.chunks[f] = owned.back().get();
                }
            }
            checkMeshersAgree(*chunk, n);
        }
    }
    SUBCASE("mixed block types merge only with their own kind")
    {
        SplitMix64 rng{7};
        for (BlockId& b : chunk->blocks()) {
            const f64 r = rng.nextUnit();
            b = r < 0.4 ? kAir : static_cast<BlockId>(1 + (rng.next() % 3));
        }
        checkMeshersAgree(*chunk);
    }
}

TEST_CASE("Outer layers give the same boundary masks as reading the neighbours")
{
    SplitMix64 rng{2024};
    std::vector<std::unique_ptr<Chunk>> owned;
    std::vector<OuterLayers> layers(kFaceCount);
    ChunkNeighbours nb;
    std::array<const OuterLayers*, kFaceCount> nbLayers{};
    for (u32 f = 0; f < kFaceCount; ++f) {
        if (f == 3) {
            continue; // one side missing: air
        }
        owned.push_back(randomChunk(0.3 + 0.4 * rng.nextUnit(), rng.next()));
        nb.chunks[f] = owned.back().get();
        layers[f] = computeOuterLayers(*owned.back());
        nbLayers[f] = &layers[f];
    }
    const BoundaryMasks expected = buildBoundaryMasks(nb);
    const BoundaryMasks actual = boundaryFromNeighbours(nbLayers);
    CHECK(actual.rows == expected.rows);

    // And on structured terrain.
    auto terrain = std::make_unique<Chunk>();
    fillTerrain(*terrain, 3, -2);
    const OuterLayers t = computeOuterLayers(*terrain);
    for (u32 f = 0; f < kFaceCount; ++f) {
        ChunkNeighbours one;
        one.chunks[f ^ 1u] = terrain.get(); // terrain is the neighbour on the opposite side
        std::array<const OuterLayers*, kFaceCount> oneLayer{};
        oneLayer[f ^ 1u] = &t;
        CHECK(boundaryFromNeighbours(oneLayer).rows == buildBoundaryMasks(one).rows);
    }
}
