// Sandbox: an endless streamed terrain to fly over. Chunks around the camera
// are generated and meshed on background threads and uploaded as they finish.
//
// Controls: left-click to capture the mouse, Esc to release it.
//           WASD move, Space/Shift up/down, Ctrl to go fast.
//
// Options:  --radius N        view distance in chunks of 32 blocks (default 96, up to 1024)
//           --detail-radius N radius of each level of detail, in its own chunks (default 12, 4..96):
//                             full detail this far, then 2x coarser voxels for each doubling
//           --lod-levels N    force N levels of detail (default: as many as --radius needs);
//                             1 = full detail all the way (radius up to 128)
//           --time H          time of day in hours (default 10; 12 = noon, night ~19-5); hold T to advance it
//           --day-length S    seconds per in-game day (default 0: the time stands still)
//           --no-shadows      sun/moon light without shadows
//           --vsync           cap to display refresh (default: uncapped)
//           --validation      force the Vulkan validation layer on (always on in Debug)
//           --no-validation   force it off
//           --frames N        quit N frames after the initial area has loaded (automated runs)
//           --screenshot F    save the last of those frames (or frame 60) to F (.bmp)
//           --fly SPEED       fly forward automatically at SPEED blocks/s once loaded
//           --no-ao           turn ambient occlusion off (for comparison)
//           --no-face-culling draw back-facing face directions too (for comparison)
//           --cpu-culling     cull chunks on the CPU instead of the GPU (for comparison)
//           --mesh-memory MiB initial GPU mesh buffer size (grows as needed; default 64)
//           --camera x,y,z,yaw,pitch   start position (yaw/pitch in degrees)
//           --assets DIR      asset folder (default: <repo>/assets)
//           --mesher M        greedy (default) or binary (one quad per block face)
//           --threads N       threads in total for background work (default: all hardware threads)

#include "engine/core/Profile.hpp"
#include "engine/core/ThreadPool.hpp"
#include "engine/platform/Window.hpp"
#include "engine/render/BlockTextures.hpp"
#include "engine/render/Camera.hpp"
#include "engine/render/Renderer.hpp"
#include "engine/render/Sky.hpp"
#include "engine/voxel/Blocks.hpp"
#include "engine/voxel/ChunkStreamer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <memory>
#include <string>
#include <vector>

using namespace engine;

namespace {

struct Options {
    i32 radius = 96;
    f32 timeOfDay = 10.0f;
    f32 dayLength = 0.0f;
    bool shadows = true;
    i32 detailRadius = 12;
    u32 lodLevels = 0; // 0 = automatic
    bool vsync = false;
    bool validation = ENGINE_ENABLE_ASSERTS != 0; // on in Debug builds
    i64 frames = -1;
    std::string screenshot;
    std::string assets = ENGINE_ASSET_DIR;
    bool greedy = true;
    bool ao = true;
    bool faceCulling = true;
    bool gpuCulling = true;
    u32 meshMemoryMiB = 64;
    u32 threads = 0; // 0 = all hardware threads
    f32 fly = 0.0f;
    f32 camera[5] = {0.0f, 90.0f, 40.0f, 0.0f, -20.0f};
};

Options parseArgs(int argc, char** argv)
{
    Options o;
    for (int i = 1; i < argc; ++i) {
        const auto arg = [&](const char* name) { return std::strcmp(argv[i], name) == 0; };
        const auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (arg("--radius")) {
            o.radius = std::clamp(std::atoi(next()), 1, 1024);
        } else if (arg("--detail-radius")) {
            o.detailRadius = std::clamp(std::atoi(next()), 4, 96);
        } else if (arg("--lod-levels")) {
            o.lodLevels = static_cast<u32>(std::clamp(std::atoi(next()), 1, static_cast<int>(voxel::kMaxLodLevels)));
        } else if (arg("--time")) {
            o.timeOfDay = std::fmod(std::fmod(static_cast<f32>(std::atof(next())), 24.0f) + 24.0f, 24.0f);
        } else if (arg("--day-length")) {
            o.dayLength = std::max(0.0f, static_cast<f32>(std::atof(next())));
        } else if (arg("--no-shadows")) {
            o.shadows = false;
        } else if (arg("--vsync")) {
            o.vsync = true;
        } else if (arg("--validation")) {
            o.validation = true;
        } else if (arg("--no-validation")) {
            o.validation = false;
        } else if (arg("--frames")) {
            o.frames = std::atoll(next());
        } else if (arg("--screenshot")) {
            o.screenshot = next();
        } else if (arg("--mesher")) {
            o.greedy = std::strcmp(next(), "binary") != 0;
        } else if (arg("--threads")) {
            o.threads = static_cast<u32>(std::clamp(std::atoi(next()), 0, 256));
        } else if (arg("--cpu-culling")) {
            o.gpuCulling = false;
        } else if (arg("--no-face-culling")) {
            o.faceCulling = false;
        } else if (arg("--mesh-memory")) {
            o.meshMemoryMiB = static_cast<u32>(std::clamp(std::atoi(next()), 1, 4096));
        } else if (arg("--no-ao")) {
            o.ao = false;
        } else if (arg("--fly")) {
            o.fly = static_cast<f32>(std::atof(next()));
        } else if (arg("--assets")) {
            o.assets = next();
        } else if (arg("--camera")) {
            std::sscanf(next(), "%f,%f,%f,%f,%f", &o.camera[0], &o.camera[1], &o.camera[2], &o.camera[3], &o.camera[4]);
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
        }
    }
    return o;
}

// How far each level of detail streams. Level L has voxels of 2^L blocks and
// streams `radius` of its own (2^L times larger) chunks. Level L takes over
// from level L - 1 at a seam radius a little inside where level L - 1 is
// guaranteed to have meshes; the last level streams as far as needed.
struct StreamingPlan {
    std::vector<i32> radii; // per level, in that level's chunks
    i32 reachChunks = 0;    // view distance, in level-0 chunks
    f32 lodSeam = 0.0f;     // blocks (RendererDesc::lodSeam)
    u32 maxChunks = 65536;  // renderer chunk slots
};

StreamingPlan planStreaming(const Options& opt)
{
    StreamingPlan plan;
    const i32 detail = std::min(opt.detailRadius, opt.radius);
    u32 levels = opt.lodLevels;
    if (levels == 0) { // as many as needed: detail * 2^(levels - 1) >= radius
        levels = 1;
        while (levels < voxel::kMaxLodLevels && (detail << (levels - 1)) < opt.radius) {
            ++levels;
        }
    }
    if (levels == 1 || detail < 4) {
        // Full detail all the way. Memory grows with the square of this.
        plan.radii = {std::min(opt.radius, 128)};
        if (opt.radius > 128) {
            std::fprintf(stderr, "Full-detail radius limited to 128 chunks (use levels of detail to see further)\n");
        }
    } else {
        plan.radii.assign(levels - 1, detail);
        // The last level: as far as asked, but at least past its own seam.
        const i32 scale = 1 << (levels - 1);
        const i32 minOuter = (detail - 3 + 1) / 2 + 2;
        i32 outer = std::max((opt.radius + scale - 1) / scale, minOuter);
        if (outer > 128) {
            outer = 128;
            std::fprintf(stderr, "Outermost level limited to 128 chunks: add levels of detail to see further\n");
        }
        plan.radii.push_back(outer);
        plan.lodSeam = static_cast<f32>(detail - 3) * 32.0f;
    }
    plan.reachChunks = plan.radii.back() << (plan.radii.size() - 1);

    // Chunk slots: about two meshes per column on this terrain; plenty of slack.
    f64 estimate = 0.0;
    for (const i32 r : plan.radii) {
        estimate += 3.15 * (r + 2) * (r + 2) * 2.5;
    }
    plan.maxChunks = static_cast<u32>(std::clamp(estimate, 65536.0, 262144.0));
    return plan;
}

void updateCamera(render::Camera& cam, const platform::InputState& in, f32 dt)
{
    using platform::Key;
    constexpr f32 kMouseSensitivity = 0.0025f;
    cam.yaw += static_cast<f32>(in.mouseDx) * kMouseSensitivity;
    cam.pitch -= static_cast<f32>(in.mouseDy) * kMouseSensitivity;
    cam.pitch = std::clamp(cam.pitch, -1.55f, 1.55f);

    const glm::vec3 flatForward = glm::normalize(glm::vec3{std::sin(cam.yaw), 0.0f, -std::cos(cam.yaw)});
    const glm::vec3 right = cam.right();
    glm::vec3 move{0.0f};
    if (in.down(Key::W))
        move += flatForward;
    if (in.down(Key::S))
        move -= flatForward;
    if (in.down(Key::D))
        move += right;
    if (in.down(Key::A))
        move -= right;
    if (in.down(Key::Space))
        move.y += 1.0f;
    if (in.down(Key::LeftShift))
        move.y -= 1.0f;
    if (glm::dot(move, move) > 0.0f) {
        const f32 speed = in.down(Key::LeftControl) ? 120.0f : 25.0f; // blocks per second
        cam.position += glm::dvec3{glm::normalize(move) * speed * dt};
    }
}

} // namespace

int main(int argc, char** argv)
{
    const Options opt = parseArgs(argc, argv);
    std::fprintf(stderr, "Engine2 sandbox, %s build%s\n", ENGINE_BUILD_CONFIG,
                 std::strcmp(ENGINE_BUILD_CONFIG, "Debug") == 0 ? " (timings are not representative)" : "");

    platform::Window window("Engine2", 1600, 900);
    render::VulkanContext ctx(window, {.appName = "Engine2 Sandbox", .validation = opt.validation});
    const StreamingPlan plan = planStreaming(opt);
    const f32 farthest = static_cast<f32>(plan.reachChunks) * 32.0f; // blocks
    render::Renderer renderer(ctx, window,
                              {
                                  .presentMode = opt.vsync ? render::PresentMode::Vsync : render::PresentMode::Uncapped,
                                  .fogDistance = farthest * 0.95f,
                                  .maxChunks = plan.maxChunks,
                                  .quadCapacity = static_cast<u64>(opt.meshMemoryMiB) * 1024 * 1024 / 8,
                                  .faceCulling = opt.faceCulling,
                                  .gpuCulling = opt.gpuCulling,
                                  .lodSeam = plan.lodSeam,
                                  .shadows = opt.shadows,
                              });
    f32 timeOfDay = opt.timeOfDay;
    renderer.setSun(render::sunDirectionAt(timeOfDay));

    using Clock = std::chrono::steady_clock;
    {
        const auto t0 = Clock::now();
        const render::BlockTextureSet textures =
            render::loadBlockTextures(voxel::blockRegistry(), std::filesystem::path(opt.assets) / "textures");
        renderer.setBlockTextures(textures);
        std::fprintf(stderr, "Textures: %u layers, %ux%u, %u mips, %.1f ms\n", textures.textures.layers,
                     textures.textures.size, textures.textures.size, textures.textures.mipLevels,
                     std::chrono::duration<f64, std::milli>(Clock::now() - t0).count());
    }
    // Background workers for generation and meshing. --threads counts the
    // main thread too, so the pool gets one fewer.
    ThreadPool pool(opt.threads == 0 ? 0u : std::max(opt.threads, 2u) - 1);
    // One streamer per level of detail. Updated finest first, so each frame
    // the nearby (most visible) work is queued ahead of the distant work.
    std::vector<std::unique_ptr<voxel::ChunkStreamer>> streamers;
    for (u32 level = 0; level < plan.radii.size(); ++level) {
        streamers.push_back(std::make_unique<voxel::ChunkStreamer>(
            pool, voxel::StreamingOptions{
                      .viewRadius = plan.radii[level],
                      .level = level,
                      .greedy = opt.greedy,
                      .ambientOcclusion = opt.ao,
                      // Fewer queued jobs for the coarse levels, so they cannot
                      // crowd out nearby work when the camera moves.
                      .maxJobsInFlight = level == 0 ? 0u : 8u * pool.concurrency(),
                      // Coarse terrain is never edited: keep only meshes.
                      .keepChunkData = level == 0,
                  }));
    }
    const auto streamStats = [&] {
        voxel::ChunkStreamer::Stats total;
        for (const auto& s : streamers) {
            const voxel::ChunkStreamer::Stats ss = s->stats();
            total.loadedColumns += ss.loadedColumns;
            total.storedChunks += ss.storedChunks;
            total.meshedChunks += ss.meshedChunks;
            total.jobsInFlight += ss.jobsInFlight;
            total.poolChunks += ss.poolChunks;
        }
        return total;
    };
    std::fprintf(stderr, "Culling: %s\n", renderer.gpuCulling() ? "GPU (compute + draw indirect count)" : "CPU");
    std::string radii;
    for (const i32 r : plan.radii) {
        radii += (radii.empty() ? "" : ", ") + std::to_string(r);
    }
    std::fprintf(stderr,
                 "Streaming: view distance %d chunks (%.0f blocks), %zu levels of detail (radii %s), %u worker "
                 "threads, %s meshing\n",
                 plan.reachChunks, static_cast<f64>(farthest), plan.radii.size(), radii.c_str(), pool.workerCount(),
                 opt.greedy ? "greedy" : "binary");

    render::Camera camera;
    camera.position = {opt.camera[0], opt.camera[1], opt.camera[2]};
    camera.yaw = glm::radians(opt.camera[3]);
    camera.pitch = glm::radians(opt.camera[4]);

    // Title-bar stats, averaged over half a second.
    auto statsStart = Clock::now();
    f64 cpuAccumMs = 0.0, gpuAccumMs = 0.0, streamAccumMs = 0.0, cullAccumMs = 0.0, waitAccumMs = 0.0;
    u64 uploadAccum = 0;
    u32 statFrames = 0;

    // Automated runs (--frames) count frames from when the initial area has
    // finished loading, so screenshots and timings do not depend on load speed.
    const auto start = Clock::now();
    bool loaded = false;
    i64 framesAfterLoad = 0;
    const i64 screenshotFrame = opt.screenshot.empty() ? -1 : (opt.frames > 0 ? opt.frames - 1 : 60);

    u32 lastGrows = 0;
    auto last = Clock::now();
    while (!window.shouldClose()) {
        if (loaded && opt.frames >= 0 && framesAfterLoad >= opt.frames) {
            break;
        }
        window.pollEvents();
        const auto now = Clock::now();
        const f32 dt = std::min(std::chrono::duration<f32>(now - last).count(), 0.1f);
        last = now;

        updateCamera(camera, window.input(), dt);
        // Time of day: runs with --day-length; T fast-forwards (an hour per second).
        const f32 hoursPerSecond = (opt.dayLength > 0.0f ? 24.0f / opt.dayLength : 0.0f) +
                                   (window.input().down(platform::Key::T) ? 1.0f : 0.0f);
        if (hoursPerSecond > 0.0f) {
            timeOfDay = std::fmod(timeOfDay + hoursPerSecond * dt, 24.0f);
            renderer.setSun(render::sunDirectionAt(timeOfDay));
        }
        if (loaded && opt.fly > 0.0f) {
            const glm::vec3 flat = glm::normalize(glm::vec3{std::sin(camera.yaw), 0.0f, -std::cos(camera.yaw)});
            camera.position += glm::dvec3{flat * opt.fly * dt};
        }

        // Streaming: schedule work around the camera, hand finished meshes to the GPU.
        const auto streamStart = Clock::now();
        bool settled = true;
        for (const auto& streamer : streamers) {
            const u32 level = streamer->level();
            streamer->update(voxel::columnAt(camera.position.x, camera.position.z, level));
            for (voxel::ChunkMeshEvent& e : streamer->takeMeshEvents()) {
                if (e.removed) {
                    renderer.removeChunkMesh(e.coord.key(level));
                } else {
                    // Blocks per chunk, horizontally and vertically.
                    const i32 n = static_cast<i32>(voxel::kChunkSize) * voxel::lodScale(level);
                    const i32 v = static_cast<i32>(voxel::kChunkSize) * voxel::lodVerticalScale(level);
                    renderer.setChunkMesh(e.coord.key(level), {e.coord.x * n, e.coord.y * v, e.coord.z * n}, e.quads,
                                          level);
                    streamer->recycle(std::move(e.quads)); // the renderer copied it: reuse the vector
                }
            }
            settled &= streamer->settled();
        }
        streamAccumMs += std::chrono::duration<f64, std::milli>(Clock::now() - streamStart).count();

        if (!loaded && settled && renderer.meshStats().pendingBytes == 0) {
            loaded = true;
            const voxel::ChunkStreamer::Stats ss = streamStats();
            const render::MeshMemoryStats ms = renderer.meshStats();
            std::fprintf(stderr,
                         "Initial area loaded in %.1f ms: %u columns (%u chunks stored), %u chunk meshes, %.2fM quads "
                         "(%.1f MiB)\n",
                         std::chrono::duration<f64, std::milli>(Clock::now() - start).count(), ss.loadedColumns,
                         ss.storedChunks, ss.meshedChunks, static_cast<f64>(ms.quads) / 1e6,
                         static_cast<f64>(ms.usedBytes) / (1024.0 * 1024.0));
        }
        if (loaded && framesAfterLoad == screenshotFrame) {
            renderer.requestScreenshot(opt.screenshot);
        }

        const auto cpuStart = Clock::now();
        const render::FrameStats stats = renderer.renderFrame(camera);
        cpuAccumMs += std::chrono::duration<f64, std::milli>(Clock::now() - cpuStart).count() - stats.waitMs;
        waitAccumMs += stats.waitMs;
        gpuAccumMs += stats.gpuMs;
        cullAccumMs += stats.cullMs;
        uploadAccum += stats.uploadedBytes;
        ++statFrames;
        if (loaded) {
            ++framesAfterLoad;
        }
        ENGINE_PROFILE_FRAME();

        const f64 elapsed = std::chrono::duration<f64>(Clock::now() - statsStart).count();
        if (elapsed >= 0.5) {
            const voxel::ChunkStreamer::Stats ss = streamStats();
            const render::MeshMemoryStats ms = renderer.meshStats();
            // render cpu excludes waiting for the GPU (shown separately as wait).
            const std::string title = std::format(
                "Engine2 | {:.0f} fps | render cpu {:.2f} ms (cpu cull {:.2f}), wait {:.2f} ms | gpu {:.2f} ms | "
                "stream {:.2f} ms | {} chunks, {} draws ({} seam, {} shadow), {:.2f}M quads | {} meshed, {} jobs, "
                "{:.1f} MB/s up, mesh mem {:.0f}/{:.0f} MiB | {:02d}:{:02d}",
                statFrames / elapsed, cpuAccumMs / statFrames, cullAccumMs / statFrames, waitAccumMs / statFrames,
                gpuAccumMs / statFrames, streamAccumMs / statFrames, stats.drawnChunks, stats.draws, stats.seamDraws,
                stats.shadowDraws, static_cast<f64>(stats.drawnQuads) / 1e6, ss.meshedChunks, ss.jobsInFlight,
                static_cast<f64>(uploadAccum) / elapsed / 1e6, static_cast<f64>(ms.usedBytes) / (1024.0 * 1024.0),
                static_cast<f64>(ms.capacityBytes) / (1024.0 * 1024.0), static_cast<int>(timeOfDay),
                static_cast<int>(std::fmod(timeOfDay, 1.0f) * 60.0f));
            window.setTitle(title.c_str());
            if (opt.frames >= 0) {
                std::fprintf(stderr, "%s\n", title.c_str());
            }
            if (ms.failed > 0) {
                std::fprintf(stderr, "Warning: %u chunk meshes did not fit (raise RendererDesc::maxChunks)\n",
                             ms.failed);
            }
            if (ms.grows != lastGrows) {
                std::fprintf(stderr, "Mesh buffer grew to %.0f MiB\n",
                             static_cast<f64>(ms.capacityBytes) / (1024.0 * 1024.0));
                lastGrows = ms.grows;
            }
            statsStart = Clock::now();
            cpuAccumMs = gpuAccumMs = streamAccumMs = cullAccumMs = waitAccumMs = 0.0;
            uploadAccum = 0;
            statFrames = 0;
        }
    }
    return 0;
}
