// Sandbox: generates and meshes a patch of world, then prints timings.
// This is where a window and renderer get added once a graphics API is chosen.

#include "engine/core/Profile.hpp"
#include "engine/voxel/Generate.hpp"
#include "engine/voxel/Mesher.hpp"

#include <chrono>
#include <cstdio>
#include <format>
#include <memory>
#include <vector>

using namespace engine;
using namespace engine::voxel;

int main()
{
    constexpr i32 kRadius = 8; // 16x16 = 256 chunks
    constexpr int kFrames = 30;

    auto chunk = std::make_unique<Chunk>();
    std::vector<PackedQuad> quads;
    quads.reserve(64 * 1024);

    using Clock = std::chrono::steady_clock;
    std::chrono::nanoseconds genTime{}, meshTime{};
    usize totalQuads = 0;

    for (int frame = 0; frame < kFrames; ++frame) {
        ENGINE_PROFILE_SCOPE("frame");
        totalQuads = 0;
        for (i32 cz = -kRadius; cz < kRadius; ++cz) {
            for (i32 cx = -kRadius; cx < kRadius; ++cx) {
                const auto t0 = Clock::now();
                fillTerrain(*chunk, cx, cz);
                const auto t1 = Clock::now();
                meshChunkBinary(*chunk, quads);
                const auto t2 = Clock::now();
                genTime += t1 - t0;
                meshTime += t2 - t1;
                totalQuads += quads.size();
            }
        }
        ENGINE_PROFILE_FRAME();
    }

    constexpr auto chunksPerFrame = static_cast<f64>(4 * kRadius * kRadius);
    const auto perChunkUs = [&](std::chrono::nanoseconds t) {
        return std::chrono::duration<f64, std::micro>(t).count() / (chunksPerFrame * kFrames);
    };

    std::fputs(std::format("chunks/frame : {}\n", static_cast<int>(chunksPerFrame)).c_str(), stdout);
    std::fputs(std::format("quads/frame  : {} ({:.1f} MiB packed)\n", totalQuads,
                           static_cast<f64>(totalQuads * sizeof(PackedQuad)) / (1024.0 * 1024.0))
                   .c_str(),
               stdout);
    std::fputs(std::format("generate     : {:.2f} us/chunk\n", perChunkUs(genTime)).c_str(), stdout);
    std::fputs(std::format("mesh         : {:.2f} us/chunk\n", perChunkUs(meshTime)).c_str(), stdout);
    return 0;
}
