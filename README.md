# Engine2

A voxel game engine in C++23, built for performance.

## Setting up Visual Studio 2022

1. Install or update to **Visual Studio 2022 17.8 or newer** (Help → Check for Updates).
2. In the **Visual Studio Installer**, choose *Modify* and enable the
   **Desktop development with C++** workload. On the right-hand side make sure these are ticked:
   - *MSVC v143 - VS 2022 C++ x64/x86 build tools (Latest)*
   - *C++ CMake tools for Windows* (brings CMake and Ninja)
   - *Windows 11 SDK* (any recent version)
   - *C++ Clang tools for Windows* (optional, for the `clangcl-release` preset)
3. Install **Git for Windows** if you don't have it. CMake uses it to download dependencies.
4. Install the **Vulkan SDK** from https://vulkan.lunarg.com/sdk/home (Windows installer, default options).
   It provides `glslc` (the shader compiler) and the validation layers. Restart Visual Studio afterwards
   so it sees the `VULKAN_SDK` environment variable.
5. Keep your GPU driver up to date: the renderer needs **Vulkan 1.3** (any desktop GPU driver from 2022 on).

There is no `.sln` file. This is a CMake project, and Visual Studio opens it directly:

1. **File → Open → Folder…** and pick the repository root.
2. Visual Studio reads `CMakePresets.json` and configures the project (dependencies download the first time).
3. Use the configuration dropdown in the toolbar to choose a preset (see below).
4. Use the **startup item** dropdown (next to the green ▶) to pick `sandbox.exe`, `engine_tests.exe`
   or `engine_bench.exe`, then press F5 (debug) or Ctrl+F5 (run).
5. **Test → Test Explorer** lists the CTest tests.

### Command line (optional)

Open the **x64 Native Tools Command Prompt for VS 2022** (not the plain Developer prompt, which may be 32-bit):

```bat
cmake --preset msvc-release
cmake --build --preset msvc-release
ctest --preset msvc-release
out\build\msvc-release\bin\engine_bench.exe
```

## Presets

| Preset            | Build type     | Use it for                                                     |
|-------------------|----------------|----------------------------------------------------------------|
| `msvc-debug`      | Debug          | Stepping through code. Asserts on. Slow.                       |
| `msvc-release`    | Release        | Shipping build: `/O2 /Ob3`, LTO (`/GL` + `/LTCG`), AVX2.       |
| `msvc-profile`    | RelWithDebInfo | **Profiling.** Same code generation as Release, plus PDBs and Tracy. |
| `msvc-asan`       | Debug          | AddressSanitizer: finds out-of-bounds and use-after-free bugs. |
| `clangcl-release` | Release        | A second optimiser to compare codegen and speed with MSVC.    |
| `linux-*`         | various        | Linux/CI builds with GCC and Clang.                           |

Build output goes to `out/build/<preset>/bin/`.

The sandbox prints its build configuration at startup. **Always measure in Release or Profile.** Debug MSVC builds can be 10–50× slower
(no inlining, checked iterators), so their timings tell you nothing.

## Performance tooling

- **Benchmarks** (`bench/`, [nanobench](https://github.com/martinus/nanobench)): run `engine_bench.exe`
  (or `engine_bench.exe mesh` / `world`) from a Release build. Each benchmark compares implementations side by side, e.g. the two meshers.
- **Tracy profiler** (`msvc-profile`): download the Tracy **v0.11.1** GUI from
  [its releases page](https://github.com/wolfpld/tracy/releases/tag/v0.11.1), launch it, then run the
  sandbox. Mark code with `ENGINE_PROFILE_SCOPE("name")` / `ENGINE_PROFILE_FUNCTION()` from
  `engine/core/Profile.hpp`. In other builds these compile to nothing.
- **Visual Studio profiler**: Debug → Performance Profiler → CPU Usage, on the `msvc-profile` build.
- **Correctness**: hand-optimised code keeps a simple reference version, and tests check that the two
  give identical output (see `tests/test_mesher.cpp`).

## Renderer

Vulkan 1.3, written for throughput:

- **Meshing** (`engine/voxel/Mesher.hpp`): visible faces are found 32 voxels at a time with bit operations,
  faces between neighbouring chunks are culled, and adjacent faces of the same block are merged into
  rectangles (greedy meshing). On the sandbox world that cuts 2.87M quads to 0.62M (4.6x).
- **Streaming** (`engine/voxel/ChunkStreamer.hpp`): the world is columns of 4 chunks (128 blocks tall).
  Columns around the camera are generated and their chunks meshed by background jobs on a thread pool
  (`engine/core/ThreadPool.hpp`), nearest first, and unloaded behind it. Chunks that are all air or all
  stone take no memory, and air chunks or buried stone chunks need no meshing. A chunk records its outer
  layers as bitmasks when generated, so meshing its neighbours never reads its (cold) memory. Chunk memory
  is pooled and reused. The renderer keeps all chunk meshes in one GPU buffer, divided up by a VMA virtual
  allocator, and uploads new meshes within a per-frame byte budget.
- **Ambient occlusion**: each face corner is darkened by the blocks touching it (two sides and the
  diagonal), stored as 2 bits per corner in the quad and interpolated across it. Greedy meshing merges
  only faces with identical occlusion, so merged quads shade exactly like separate faces, and each quad's
  triangle split is chosen to avoid diagonal streaks. Corners at chunk edges read the full one-voxel shell
  from all 26 neighbours. Off with `--no-ao`.
- **Long view distances**: everything is drawn relative to the camera (integer block + fraction, camera
  position in double precision), so precision is the same a million blocks from the origin. Each chunk's
  quads are grouped by face direction and only the directions that can face the camera are drawn (about
  27% fewer quads on terrain). The mesh buffer grows on demand. `--radius` goes up to 96.
- **GPU-driven culling**: a compute shader tests every chunk (distance, frustum, face directions) and writes
  the draw commands; one `vkCmdDrawIndexedIndirectCount` draws them, so the CPU never walks the chunk list.
  Falls back to CPU culling (`--cpu-culling`) on GPUs without draw-indirect-count. Both paths produce
  pixel-identical images.
- **Levels of detail** for distant terrain: level L is the same world with voxels of 2^L blocks, so a
  level-L chunk covers 2^L times the distance with the same 32^3 cells, mesher, ambient occlusion and
  culling. Each level streams `--detail-radius` (default 12) of its own chunks around the camera, and as
  many levels are used as `--radius` (the view distance, up to 1024 chunks = 32,768 blocks) needs.
  Level L takes over from level L-1 at a horizontal seam radius, tested per pixel (horizontal distance
  only grows along a view ray, so there are no gaps); the finer level is drawn a little past the seam
  with a tiny depth offset in its favour. Only coarse chunks that cross a seam use the discarding
  pipeline; chunks entirely inside the seam are culled. Coarse levels release each chunk's voxels as
  soon as it is meshed (only the mesh and its outer layers are kept), so memory stays small.
- **Explicit AVX2** in the hot loops (voxel bitmasks, terrain layers, face masks), with plain C++
  fallbacks, so speed does not depend on what a compiler's auto-vectoriser happens to do.
- **One draw call for the whole world**: every visible chunk's draws are indirect commands in one buffer.
- **Reversed-Z, infinite far plane** with a 32-bit float depth buffer: even depth precision at any distance.
- **volk** loads Vulkan functions directly from the driver, skipping the loader's dispatch on each call.
- **Dynamic rendering + synchronization2**: no render pass or framebuffer objects.
- **Block textures** live in `assets/textures/*.png` (see the README there), packed into one texture
  array with CPU-built mipmaps (filtered in linear space). Sampling is nearest-neighbour up close and
  trilinear + 16x anisotropic at a distance. Which texture goes on which face is set per block in
  `engine/include/engine/voxel/Blocks.hpp`.
- Shaders are compiled at build time and embedded in the executable.

Sandbox controls: left-click to capture the mouse, Esc to release; WASD to move, Space/Shift for
up/down, Ctrl for speed. Frame rate, CPU and GPU times are shown in the title bar.

Sandbox options: `--radius N` (view distance in chunks, default 96, up to 1024), `--detail-radius N`
(radius of each level of detail, default 12), `--lod-levels N` (force a number of levels; 1 = full detail
only, radius up to 128), `--vsync`, `--validation` / `--no-validation`,
`--camera x,y,z,yawDeg,pitchDeg`, `--assets DIR`, `--mesher greedy|binary`, `--no-ao`, `--no-face-culling`, `--cpu-culling`, `--mesh-memory MiB`, `--threads N`,
`--fly SPEED` (fly forward automatically, blocks/s), `--frames N` (quit N frames after the initial area
has loaded), `--screenshot file.bmp`. The title bar shows frame, CPU (excluding time blocked on the
GPU, shown as `wait`), GPU and streaming times, chunks and draws (`seam`: coarse chunks crossing a seam),
chunks meshed, background jobs, upload rate and mesh memory.

Validation is on by default in Debug builds. For deeper checks, set these environment variables
(in Visual Studio: right-click `sandbox.exe` → *Add Debug Configuration* → `"env"`):
`VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true` and `VK_KHRONOS_VALIDATION_VALIDATE_BEST_PRACTICES=true`.

## Layout

```
cmake/            Compiler flags (CompilerOptions.cmake), dependencies (Dependencies.cmake)
engine/include/   Public headers: engine/core/*, engine/voxel/*
engine/src/       Engine implementation (static library, no graphics or OS code)
renderer/         Vulkan renderer, window/input (GLFW), shaders
apps/sandbox/     Executable for trying things out
tests/            Unit tests (doctest)
bench/            Micro-benchmarks (nanobench)
```

Dependencies are fetched by CMake at configure time and pinned to exact commits. Nothing is vendored
and nothing needs installing by hand.

## CMake options

| Option                      | Default | Effect                                         |
|-----------------------------|---------|------------------------------------------------|
| `ENGINE_ENABLE_AVX2`        | ON      | `/arch:AVX2` / `-march=x86-64-v3`              |
| `ENGINE_ENABLE_LTO`         | ON      | Link-time optimisation in Release/Profile      |
| `ENGINE_ENABLE_TRACY`       | OFF     | Tracy client (ON in `msvc-profile`)            |
| `ENGINE_ENABLE_SANITIZERS`  | OFF     | ASan (+UBSan on GCC/Clang)                     |
| `ENGINE_WARNINGS_AS_ERRORS` | OFF     | `/WX` / `-Werror` (CI turns it on)             |
| `ENGINE_BUILD_RENDERER`     | ON      | Vulkan renderer + windowed sandbox             |
