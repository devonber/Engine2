#pragma once

#include "engine/voxel/Mesher.hpp"

#include <vector>

namespace engine {
class ThreadPool;
}

namespace engine::voxel {

struct WorldBuildOptions {
    i32 radius = 12;           // world is (2 * radius)^2 chunks, one chunk tall, centred on 0
    bool greedy = true;        // greedy meshing (else one quad per block face)
    bool borderCulling = true; // cull faces against neighbouring chunks
};

// One chunk's slice of WorldMesh::quads.
struct ChunkMeshRange {
    i32 chunkX = 0;
    i32 chunkZ = 0;
    u32 firstQuad = 0;
    u32 quadCount = 0;
};

struct WorldMesh {
    std::vector<PackedQuad> quads;
    std::vector<ChunkMeshRange> chunks;
    f64 generateMs = 0.0;
    f64 meshMs = 0.0;
};

// Generates the sandbox terrain and meshes every chunk against its
// neighbours. Both phases run on `pool` if given (else on this thread).
// The output is identical for any thread count.
[[nodiscard]] WorldMesh buildTerrainWorld(const WorldBuildOptions& options, ThreadPool* pool);

} // namespace engine::voxel
