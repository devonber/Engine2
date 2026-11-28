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

// Split greedy rectangles back into one unit quad per block face (each with
// the rectangle's occlusion: merging only joins faces whose levels match).
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
                unit.push_back(PackedQuad::make(c[0], c[1], c[2], q.face(), q.block(), 1, 1, q.aoBits()));
            }
        }
    }
    return unit;
}

// Naive (reads neighbour chunks directly) is the reference for binary and
// greedy (which read the neighbour shell built from the same chunks).
void checkMeshersAgree(const Chunk& chunk, const ChunkNeighbourhood& neighbours = {})
{
    const NeighbourShell shell = buildNeighbourShell(neighbours);
    for (const bool ao : {false, true}) {
        CAPTURE(ao);
        std::vector<PackedQuad> naive, binary, greedy;
        meshChunkNaive(chunk, neighbours, naive, ao);
        meshChunkBinary(chunk, shell, binary, ao);
        meshChunkGreedy(chunk, shell, greedy, ao);

        const auto reference = sorted(naive);
        REQUIRE(binary.size() == naive.size());
        CHECK(sorted(binary) == reference);

        // Same faces covered exactly once (no gaps, no overlaps), same occlusion.
        const auto greedyFaces = expand(greedy);
        REQUIRE(greedyFaces.size() == naive.size());
        CHECK(sorted(greedyFaces) == reference);
        CHECK(greedy.size() <= naive.size());
    }
}

std::unique_ptr<Chunk> randomChunk(f64 density, u64 seed)
{
    auto c = std::make_unique<Chunk>();
    fillRandom(*c, density, seed);
    return c;
}

// The quad for one face in a mesh (unit faces only), or nullptr.
const PackedQuad* findFace(const std::vector<PackedQuad>& quads, u32 x, u32 y, u32 z, Face face)
{
    for (const PackedQuad& q : quads) {
        if (q.x() == x && q.y() == y && q.z() == z && q.face() == face) {
            return &q;
        }
    }
    return nullptr;
}

} // namespace

TEST_CASE("PackedQuad round-trips its fields")
{
    constexpr auto q = PackedQuad::make(31, 17, 3, Face::NegZ, 0xBEEF);
    static_assert(q.x() == 31 && q.y() == 17 && q.z() == 3);
    static_assert(q.face() == Face::NegZ && q.block() == 0xBEEF);
    static_assert(q.width() == 1 && q.height() == 1);
    static_assert(q.aoBits() == PackedQuad::kNoOcclusion); // open by default

    constexpr auto big = PackedQuad::make(0, 5, 0, Face::PosY, 7, 32, 32, 0b11'10'01'00);
    static_assert(big.width() == 32 && big.height() == 32 && big.y() == 5 && big.block() == 7);
    static_assert(big.ao(0) == 0 && big.ao(1) == 1 && big.ao(2) == 2 && big.ao(3) == 3);
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
    const NeighbourShell air{};

    // Isolated shapes have nothing in front of any face: occlusion changes nothing.
    for (const bool ao : {false, true}) {
        CAPTURE(ao);
        chunk->fill(kAir);

        meshChunkBinary(*chunk, air, quads, ao);
        CHECK(quads.empty());
        meshChunkGreedy(*chunk, air, quads, ao);
        CHECK(quads.empty());

        chunk->set(10, 10, 10, 1);
        meshChunkBinary(*chunk, air, quads, ao);
        CHECK(quads.size() == 6);
        meshChunkGreedy(*chunk, air, quads, ao);
        CHECK(quads.size() == 6);

        chunk->set(11, 10, 10, 1); // two adjacent blocks hide the shared faces
        meshChunkBinary(*chunk, air, quads, ao);
        CHECK(quads.size() == 10);
        meshChunkGreedy(*chunk, air, quads, ao);
        CHECK(quads.size() == 6); // the four long sides merge into 1x2 quads

        chunk->fill(1); // full chunk: one quad per side when greedy
        meshChunkBinary(*chunk, air, quads, ao);
        CHECK(quads.size() == 6 * kChunkArea);
        meshChunkGreedy(*chunk, air, quads, ao);
        REQUIRE(quads.size() == 6);
        for (const auto& q : quads) {
            CHECK(q.width() == 32);
            CHECK(q.height() == 32);
        }

        NeighbourShell solid; // solid neighbours hide every border face
        solid.setAllSolid();
        meshChunkBinary(*chunk, solid, quads, ao);
        CHECK(quads.empty());
        meshChunkGreedy(*chunk, solid, quads, ao);
        CHECK(quads.empty());
    }

    SUBCASE("different blocks do not merge")
    {
        chunk->fill(kAir);
        for (u32 x = 0; x < kChunkSize; ++x) {
            chunk->set(x, 0, 0, x % 2 == 0 ? 1 : 2);
        }
        meshChunkGreedy(*chunk, air, quads);
        // Each block keeps its own top, bottom and two side faces; only the ends are shared.
        CHECK(quads.size() == 32 * 4 + 2);
    }
}

TEST_CASE("Ambient occlusion levels")
{
    // A stone floor at y = 0 with a few blocks standing on it.
    auto chunk = std::make_unique<Chunk>();
    for (u32 z = 0; z < kChunkSize; ++z) {
        for (u32 x = 0; x < kChunkSize; ++x) {
            chunk->set(x, 0, z, 1);
        }
    }
    chunk->set(10, 1, 10, 1); // a lone block
    chunk->set(21, 1, 22, 1); // an inside corner at (22, 0, 22): blocks to its -X and -Z
    chunk->set(22, 1, 21, 1);

    std::vector<PackedQuad> quads;
    meshChunkNaive(*chunk, {}, quads);

    // Top face of a floor block (+Y): U = z, V = x, corner = cu + 2 * cv.
    SUBCASE("open floor is unoccluded")
    {
        const PackedQuad* far = findFace(quads, 2, 0, 2, Face::PosY);
        REQUIRE(far != nullptr);
        CHECK(far->aoBits() == 0xFF);
    }
    SUBCASE("floor next to a block darkens the two corners touching it")
    {
        // Floor at x = 11, next to the block at x = 10: its corners with
        // cv = 0 (the x = 11 edge, shared with the block's side) touch it.
        const PackedQuad* next = findFace(quads, 11, 0, 10, Face::PosY);
        REQUIRE(next != nullptr);
        CHECK(next->ao(0) == 2); // one side solid
        CHECK(next->ao(1) == 2);
        CHECK(next->ao(2) == 3);
        CHECK(next->ao(3) == 3);
        // Diagonal to the block: only the diagonal voxel of one corner.
        const PackedQuad* diag = findFace(quads, 11, 0, 11, Face::PosY);
        REQUIRE(diag != nullptr);
        CHECK(diag->ao(0) == 2); // corner (cu=0, cv=0) = towards (x-1, z-1): the block is its diagonal
        CHECK(diag->ao(1) == 3);
        CHECK(diag->ao(2) == 3);
        CHECK(diag->ao(3) == 3);
    }
    SUBCASE("an inside corner is fully occluded")
    {
        const PackedQuad* inside = findFace(quads, 22, 0, 22, Face::PosY);
        REQUIRE(inside != nullptr);
        CHECK(inside->ao(0) == 0); // both sides solid
    }
    SUBCASE("block sides are darkened where they meet the floor")
    {
        // +X side of the lone block: U = y, V = z. Its bottom corners (cu = 0)
        // have the floor in front of them both straight below and diagonally
        // (one side + the diagonal solid): level 1.
        const PackedQuad* side = findFace(quads, 10, 1, 10, Face::PosX);
        REQUIRE(side != nullptr);
        CHECK(side->ao(0) == 1);
        CHECK(side->ao(2) == 1);
        CHECK(side->ao(1) == 3);
        CHECK(side->ao(3) == 3);
    }

    checkMeshersAgree(*chunk);
}

TEST_CASE("Ambient occlusion reads diagonal neighbour chunks at chunk corners")
{
    // Floor along this chunk's top layer; a single block sits in the chunk
    // diagonally above-and-beside (+X, +Y). It touches the top face of the
    // floor voxel at (31, 31, 5) only through the shell's edge data.
    auto chunk = std::make_unique<Chunk>();
    for (u32 z = 0; z < kChunkSize; ++z) {
        for (u32 x = 0; x < kChunkSize; ++x) {
            chunk->set(x, 31, z, 1);
        }
    }
    auto diagonal = std::make_unique<Chunk>();
    diagonal->set(0, 0, 5, 1);
    ChunkNeighbourhood nb;
    nb.set(+1, +1, 0, diagonal.get());

    std::vector<PackedQuad> naive;
    meshChunkNaive(*chunk, nb, naive);
    const PackedQuad* face = findFace(naive, 31, 31, 5, Face::PosY);
    REQUIRE(face != nullptr);
    CHECK(face->aoBits() != 0xFF); // occluded by the block in the diagonal chunk

    checkMeshersAgree(*chunk, nb);
}

TEST_CASE("Fast meshers match the naive reference")
{
    auto chunk = std::make_unique<Chunk>();

    SUBCASE("terrain, no neighbours")
    {
        for (i32 c = -3; c <= 3; ++c) {
            fillTerrain(*chunk, c, 1, c * 7);
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
    SUBCASE("terrain with all 26 terrain neighbours")
    {
        constexpr i32 cx = 2, cy = 1, cz = 5;
        fillTerrain(*chunk, cx, cy, cz);
        std::vector<std::unique_ptr<Chunk>> owned;
        ChunkNeighbourhood nb;
        for (i32 dz = -1; dz <= 1; ++dz) {
            for (i32 dy = -1; dy <= 1; ++dy) {
                for (i32 dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0 && dz == 0) {
                        continue;
                    }
                    owned.push_back(std::make_unique<Chunk>());
                    fillTerrain(*owned.back(), cx + dx, cy + dy, cz + dz);
                    nb.set(dx, dy, dz, owned.back().get());
                }
            }
        }
        checkMeshersAgree(*chunk, nb);
    }
    SUBCASE("random chunk with random neighbours, some missing")
    {
        SplitMix64 rng{99};
        for (int trial = 0; trial < 6; ++trial) {
            CAPTURE(trial);
            fillRandom(*chunk, 0.2 + 0.6 * rng.nextUnit(), rng.next());
            std::vector<std::unique_ptr<Chunk>> owned;
            ChunkNeighbourhood nb;
            for (usize i = 0; i < kNeighbourhoodSize; ++i) {
                if (i != neighbourIndex(0, 0, 0) && rng.nextUnit() < 0.75) {
                    owned.push_back(randomChunk(rng.nextUnit(), rng.next()));
                    nb.chunks[i] = owned.back().get();
                }
            }
            checkMeshersAgree(*chunk, nb);
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

TEST_CASE("Shell from outer layers equals the shell read from the neighbours")
{
    SplitMix64 rng{2024};
    std::vector<std::unique_ptr<Chunk>> owned;
    std::vector<OuterLayers> layers(kNeighbourhoodSize);
    ChunkNeighbourhood nb;
    std::array<const OuterLayers*, kNeighbourhoodSize> nbLayers{};
    for (usize i = 0; i < kNeighbourhoodSize; ++i) {
        if (i == neighbourIndex(0, 0, 0) || i % 5 == 3) {
            continue; // the chunk itself, and a few missing neighbours (air)
        }
        owned.push_back(randomChunk(0.3 + 0.4 * rng.nextUnit(), rng.next()));
        nb.chunks[i] = owned.back().get();
        layers[i] = computeOuterLayers(*owned.back());
        nbLayers[i] = &layers[i];
    }
    const NeighbourShell expected = buildNeighbourShell(nb);
    const NeighbourShell actual = shellFromOuterLayers(nbLayers);
    CHECK(actual.faces.rows == expected.faces.rows);
    CHECK(actual.edges == expected.edges);
    CHECK(actual.corners == expected.corners);

    // Uniformly solid neighbours: everything set.
    std::array<const OuterLayers*, kNeighbourhoodSize> allSolid{};
    allSolid.fill(&solidOuterLayers());
    NeighbourShell solid;
    solid.setAllSolid();
    const NeighbourShell fromSolid = shellFromOuterLayers(allSolid);
    CHECK(fromSolid.faces.rows == solid.faces.rows);
    CHECK(fromSolid.edges == solid.edges);
    CHECK(fromSolid.corners == solid.corners);
}
