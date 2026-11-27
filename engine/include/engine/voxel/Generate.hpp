#pragma once

#include "engine/voxel/Chunk.hpp"

namespace engine::voxel {

// Simple procedural fills for tests, benchmarks and the sandbox.

// Rolling sine-wave hills, stone under grass. Typical surface-chunk workload.
void fillTerrain(Chunk& chunk, i32 chunkX, i32 chunkZ);

// Each voxel is solid with probability `density`. density=0.5 is the
// worst case for face count.
void fillRandom(Chunk& chunk, f64 density, u64 seed);

} // namespace engine::voxel
