// Sandbox: an endless streamed terrain to fly over. Chunks around the camera
// are generated and meshed on background threads and uploaded as they finish.
//
// Controls: left-click to capture the mouse, Esc to release it.
//           WASD move, Space/Shift up/down, Ctrl to go fast.
//
// Options:  --radius N        view radius in chunks (default 12)
//           --vsync           cap to display refresh (default: uncapped)
//           --validation      force the Vulkan validation layer on (always on in Debug)
//           --no-validation   force it off
//           --frames N        quit N frames after the initial area has loaded (automated runs)
//           --screenshot F    save the last of those frames (or frame 60) to F (.bmp)
//           --fly SPEED       fly forward automatically at SPEED blocks/s once loaded
//           --no-ao           turn ambient occlusion off (for comparison)
//           --no-face-culling draw back-facing face directions too (for comparison)
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
#include "engine/voxel/Blocks.hpp"
#include "engine/voxel/ChunkStreamer.hpp"

#include <algorithm>
#include <chrono>
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
    i32 radius = 12;
    bool vsync = false;
    bool validation = ENGINE_ENABLE_ASSERTS != 0; // on in Debug builds
    i64 frames = -1;
    std::string screenshot;
    std::string assets = ENGINE_ASSET_DIR;
    bool greedy = true;
    bool ao = true;
    bool faceCulling = true;
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
            o.radius = std::clamp(std::atoi(next()), 1, 96);
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
    render::Renderer renderer(ctx, window,
                              {
                                  .presentMode = opt.vsync ? render::PresentMode::Vsync : render::PresentMode::Uncapped,
                                  .fogDistance = static_cast<f32>(opt.radius) * 32.0f * 0.95f,
                                  .quadCapacity = static_cast<u64>(opt.meshMemoryMiB) * 1024 * 1024 / 8,
                                  .faceCulling = opt.faceCulling,
                              });

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
    voxel::ChunkStreamer streamer(pool, {.viewRadius = opt.radius, .greedy = opt.greedy, .ambientOcclusion = opt.ao});
    std::fprintf(stderr, "Streaming: view radius %d chunks, %u worker threads, %s meshing\n", opt.radius,
                 pool.workerCount(), opt.greedy ? "greedy" : "binary");

    render::Camera camera;
    camera.position = {opt.camera[0], opt.camera[1], opt.camera[2]};
    camera.yaw = glm::radians(opt.camera[3]);
    camera.pitch = glm::radians(opt.camera[4]);

    // Title-bar stats, averaged over half a second.
    auto statsStart = Clock::now();
    f64 cpuAccumMs = 0.0, gpuAccumMs = 0.0, streamAccumMs = 0.0, cullAccumMs = 0.0;
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
        if (loaded && opt.fly > 0.0f) {
            const glm::vec3 flat = glm::normalize(glm::vec3{std::sin(camera.yaw), 0.0f, -std::cos(camera.yaw)});
            camera.position += glm::dvec3{flat * opt.fly * dt};
        }

        // Streaming: schedule work around the camera, hand finished meshes to the GPU.
        const auto streamStart = Clock::now();
        const voxel::ColumnCoord centre = voxel::columnAt(camera.position.x, camera.position.z);
        streamer.update(centre);
        for (voxel::ChunkMeshEvent& e : streamer.takeMeshEvents()) {
            if (e.removed) {
                renderer.removeChunkMesh(e.coord.key());
            } else {
                constexpr i32 n = static_cast<i32>(voxel::kChunkSize);
                renderer.setChunkMesh(e.coord.key(), {e.coord.x * n, e.coord.y * n, e.coord.z * n}, e.quads);
                streamer.recycle(std::move(e.quads)); // the renderer copied it: reuse the vector
            }
        }
        streamAccumMs += std::chrono::duration<f64, std::milli>(Clock::now() - streamStart).count();

        if (!loaded && streamer.settled() && renderer.meshStats().pendingBytes == 0) {
            loaded = true;
            const voxel::ChunkStreamer::Stats ss = streamer.stats();
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
        cpuAccumMs += std::chrono::duration<f64, std::milli>(Clock::now() - cpuStart).count();
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
            const voxel::ChunkStreamer::Stats ss = streamer.stats();
            const render::MeshMemoryStats ms = renderer.meshStats();
            const std::string title = std::format(
                "Engine2 | {:.0f} fps | render cpu {:.2f} ms (cull {:.2f}) | gpu {:.2f} ms | stream {:.2f} ms | "
                "{} chunks, {} draws, {:.2f}M quads | {} meshed, {} jobs, {:.1f} MB/s up, mesh mem {:.0f}/{:.0f} MiB",
                statFrames / elapsed, cpuAccumMs / statFrames, cullAccumMs / statFrames, gpuAccumMs / statFrames,
                streamAccumMs / statFrames, stats.drawnChunks, stats.draws, static_cast<f64>(stats.drawnQuads) / 1e6,
                ss.meshedChunks, ss.jobsInFlight, static_cast<f64>(uploadAccum) / elapsed / 1e6,
                static_cast<f64>(ms.usedBytes) / (1024.0 * 1024.0),
                static_cast<f64>(ms.capacityBytes) / (1024.0 * 1024.0));
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
            cpuAccumMs = gpuAccumMs = streamAccumMs = cullAccumMs = 0.0;
            uploadAccum = 0;
            statFrames = 0;
        }
    }
    return 0;
}
