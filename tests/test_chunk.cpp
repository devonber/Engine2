#include "engine/voxel/Chunk.hpp"

#include <doctest/doctest.h>

#include <memory>

using namespace engine;
using namespace engine::voxel;

TEST_CASE("voxelIndex covers the volume exactly once")
{
    static_assert(voxelIndex(0, 0, 0) == 0);
    static_assert(voxelIndex(kChunkSize - 1, kChunkSize - 1, kChunkSize - 1) == kChunkVolume - 1);
    static_assert(voxelIndex(1, 0, 0) == 1);          // x is fastest
    static_assert(voxelIndex(0, 0, 1) == kChunkSize); // then z
    static_assert(voxelIndex(0, 1, 0) == kChunkArea); // then y
}

TEST_CASE("Chunk starts empty and stores blocks")
{
    auto chunk = std::make_unique<Chunk>();
    CHECK(chunk->get(5, 6, 7) == kAir);
    chunk->set(5, 6, 7, 42);
    CHECK(chunk->get(5, 6, 7) == 42);
    CHECK(chunk->blocks()[voxelIndex(5, 6, 7)] == 42);

    chunk->fill(3);
    CHECK(chunk->get(0, 0, 0) == 3);
    CHECK(chunk->get(31, 31, 31) == 3);
}
