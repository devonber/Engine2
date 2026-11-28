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
- **Explicit AVX2** in the hot loops (voxel bitmasks, terrain layers, face masks), with plain C++
  fallbacks, so speed does not depend on what a compiler's auto-vectoriser happens to do.
- **One draw call for the whole world**: chunks are frustum-culled on the CPU, which writes one
  `vkCmdDrawIndexedIndirect` command per visible chunk.
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

Sandbox options: `--radius N` (view radius in chunks), `--vsync`, `--validation` / `--no-validation`,
`--camera x,y,z,yawDeg,pitchDeg`, `--assets DIR`, `--mesher greedy|binary`, `--no-ao`, `--threads N`,
`--fly SPEED` (fly forward automatically, blocks/s), `--frames N` (quit N frames after the initial area
has loaded), `--screenshot file.bmp`. The title bar shows frame, CPU, GPU and streaming times, chunks
drawn and meshed, background jobs, upload rate and mesh memory.

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
