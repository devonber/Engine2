// Sandbox: generates a patch of terrain, meshes it, and flies a camera over it.
//
// Controls: left-click to capture the mouse, Esc to release it.
//           WASD move, Space/Shift up/down, Ctrl to go fast.
//
// Options:  --radius N        world radius in chunks (default 12 -> 24x24 chunks)
//           --vsync           cap to display refresh (default: uncapped)
//           --validation      force the Vulkan validation layer on (always on in Debug)
//           --no-validation   force it off
//           --frames N        quit after N frames (for automated runs)
//           --screenshot F    save frame N-1 (needs --frames) or frame 60 to F (.bmp)
//           --camera x,y,z,yaw,pitch   start position (yaw/pitch in degrees)
//           --assets DIR      asset folder (default: <repo>/assets)
//           --mesher M        greedy (default) or binary (one quad per block face)
//           --no-border-culling   draw faces between chunks (for comparison)
//           --threads N       threads for world building (default: all; 1 = main thread only)

#include "engine/core/Profile.hpp"
#include "engine/core/ThreadPool.hpp"
#include "engine/platform/Window.hpp"
#include "engine/render/BlockTextures.hpp"
#include "engine/render/Camera.hpp"
#include "engine/render/Renderer.hpp"
#include "engine/voxel/Blocks.hpp"
#include "engine/voxel/World.hpp"

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
    bool borderCulling = true;
    u32 threads = 0; // 0 = all hardware threads
    f32 camera[5] = {0.0f, 45.0f, 40.0f, 0.0f, -20.0f};
};

Options parseArgs(int argc, char** argv)
{
    Options o;
    for (int i = 1; i < argc; ++i) {
        const auto arg = [&](const char* name) { return std::strcmp(argv[i], name) == 0; };
        const auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (arg("--radius")) {
            o.radius = std::clamp(std::atoi(next()), 1, 64);
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
        } else if (arg("--no-border-culling")) {
            o.borderCulling = false;
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
        cam.position += glm::normalize(move) * speed * dt;
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
    {
        const auto t0 = Clock::now();
        // 1 thread = everything on the main thread (no pool) for comparison.
        std::unique_ptr<ThreadPool> pool;
        if (opt.threads != 1) {
            pool = std::make_unique<ThreadPool>(opt.threads == 0 ? 0u : opt.threads - 1);
        }
        const voxel::WorldMesh world = voxel::buildTerrainWorld(
            {.radius = opt.radius, .greedy = opt.greedy, .borderCulling = opt.borderCulling}, pool.get());

        std::vector<render::ChunkDraw> draws;
        draws.reserve(world.chunks.size());
        constexpr i32 n = static_cast<i32>(voxel::kChunkSize);
        for (const voxel::ChunkMeshRange& c : world.chunks) {
            draws.push_back({c.chunkX * n, 0, c.chunkZ * n, c.firstQuad, c.quadCount});
        }
        const auto t1 = Clock::now();
        renderer.uploadWorld(world.quads, draws);
        const auto t2 = Clock::now();
        std::fprintf(stderr,
                     "World (%s mesher, border culling %s, %u threads): %zu chunks, %zu quads (%.1f MiB)\n"
                     "  generate %.1f ms, mesh %.1f ms, total build %.1f ms, upload %.1f ms\n",
                     opt.greedy ? "greedy" : "binary", opt.borderCulling ? "on" : "off",
                     pool ? pool->concurrency() : 1u, world.chunks.size(), world.quads.size(),
                     static_cast<f64>(world.quads.size() * sizeof(voxel::PackedQuad)) / (1024.0 * 1024.0),
                     world.generateMs, world.meshMs, std::chrono::duration<f64, std::milli>(t1 - t0).count(),
                     std::chrono::duration<f64, std::milli>(t2 - t1).count());
    }

    render::Camera camera;
    camera.position = {opt.camera[0], opt.camera[1], opt.camera[2]};
    camera.yaw = glm::radians(opt.camera[3]);
    camera.pitch = glm::radians(opt.camera[4]);

    const i64 screenshotFrame = opt.screenshot.empty() ? -1 : (opt.frames > 0 ? opt.frames - 1 : 60);

    // Title-bar stats, averaged over half a second.
    auto statsStart = Clock::now();
    f64 cpuAccumMs = 0.0, gpuAccumMs = 0.0;
    u32 statFrames = 0;

    auto last = Clock::now();
    for (i64 frame = 0; !window.shouldClose(); ++frame) {
        if (opt.frames >= 0 && frame >= opt.frames) {
            break;
        }
        window.pollEvents();
        const auto now = Clock::now();
        const f32 dt = std::min(std::chrono::duration<f32>(now - last).count(), 0.1f);
        last = now;

        updateCamera(camera, window.input(), dt);
        if (frame == screenshotFrame) {
            renderer.requestScreenshot(opt.screenshot);
        }

        const auto cpuStart = Clock::now();
        const render::FrameStats stats = renderer.renderFrame(camera);
        cpuAccumMs += std::chrono::duration<f64, std::milli>(Clock::now() - cpuStart).count();
        gpuAccumMs += stats.gpuMs;
        ++statFrames;
        ENGINE_PROFILE_FRAME();

        const f64 elapsed = std::chrono::duration<f64>(Clock::now() - statsStart).count();
        if (elapsed >= 0.5) {
            const std::string title =
                std::format("Engine2 | {:.0f} fps | render cpu {:.2f} ms | gpu {:.2f} ms | {} chunks | {:.2f}M quads",
                            statFrames / elapsed, cpuAccumMs / statFrames, gpuAccumMs / statFrames, stats.drawnChunks,
                            static_cast<f64>(stats.drawnQuads) / 1e6);
            window.setTitle(title.c_str());
            if (opt.frames >= 0) {
                std::fprintf(stderr, "%s\n", title.c_str());
            }
            statsStart = Clock::now();
            cpuAccumMs = gpuAccumMs = 0.0;
            statFrames = 0;
        }
    }
    return 0;
}
