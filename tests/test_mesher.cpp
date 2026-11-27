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

void checkMeshersAgree(const Chunk& chunk)
{
    std::vector<PackedQuad> naive, binary;
    meshChunkNaive(chunk, naive);
    meshChunkBinary(chunk, binary);
    REQUIRE(naive.size() == binary.size());
    CHECK(sorted(naive) == sorted(binary));
}

} // namespace

TEST_CASE("PackedQuad round-trips its fields")
{
    constexpr auto q = PackedQuad::make(31, 17, 3, Face::NegZ, 0xBEEF);
    static_assert(q.x() == 31 && q.y() == 17 && q.z() == 3);
    static_assert(q.face() == Face::NegZ && q.block() == 0xBEEF);
}

TEST_CASE("Meshing simple shapes")
{
    auto chunk = std::make_unique<Chunk>();
    std::vector<PackedQuad> quads;

    SUBCASE("empty chunk has no faces")
    {
        meshChunkBinary(*chunk, quads);
        CHECK(quads.empty());
    }
    SUBCASE("single block has 6 faces")
    {
        chunk->set(10, 10, 10, 1);
        meshChunkBinary(*chunk, quads);
        CHECK(quads.size() == 6);
        for (const auto& q : quads) {
            CHECK(q.block() == 1);
            CHECK(q.x() == 10);
        }
    }
    SUBCASE("two adjacent blocks hide the shared faces")
    {
        chunk->set(10, 10, 10, 1);
        chunk->set(11, 10, 10, 1);
        meshChunkBinary(*chunk, quads);
        CHECK(quads.size() == 10);
    }
    SUBCASE("full chunk only has its outer shell")
    {
        chunk->fill(1);
        meshChunkBinary(*chunk, quads);
        CHECK(quads.size() == 6 * kChunkArea);
    }
    SUBCASE("blocks on every border")
    {
        chunk->set(0, 0, 0, 1);
        chunk->set(31, 31, 31, 2);
        meshChunkBinary(*chunk, quads);
        CHECK(quads.size() == 12);
    }
}

TEST_CASE("Binary mesher matches the naive reference")
{
    auto chunk = std::make_unique<Chunk>();

    SUBCASE("terrain")
    {
        for (i32 c = -3; c <= 3; ++c) {
            fillTerrain(*chunk, c, c * 7);
            checkMeshersAgree(*chunk);
        }
    }
    SUBCASE("random densities")
    {
        for (const f64 density : {0.01, 0.1, 0.5, 0.9, 0.99}) {
            CAPTURE(density);
            fillRandom(*chunk, density, 1234);
            checkMeshersAgree(*chunk);
        }
    }
}
