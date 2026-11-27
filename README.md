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

**Always measure in Release or Profile.** Debug MSVC builds can be 10–50× slower
(no inlining, checked iterators), so their timings tell you nothing.

## Performance tooling

- **Benchmarks** (`bench/`, [nanobench](https://github.com/martinus/nanobench)): run `engine_bench.exe`
  from a Release build. Each benchmark compares implementations side by side, e.g. the two meshers.
- **Tracy profiler** (`msvc-profile`): download the Tracy **v0.11.1** GUI from
  [its releases page](https://github.com/wolfpld/tracy/releases/tag/v0.11.1), launch it, then run the
  sandbox. Mark code with `ENGINE_PROFILE_SCOPE("name")` / `ENGINE_PROFILE_FUNCTION()` from
  `engine/core/Profile.hpp`. In other builds these compile to nothing.
- **Visual Studio profiler**: Debug → Performance Profiler → CPU Usage, on the `msvc-profile` build.
- **Correctness**: hand-optimised code keeps a simple reference version, and tests check that the two
  give identical output (see `tests/test_mesher.cpp`).

## Layout

```
cmake/            Compiler flags (CompilerOptions.cmake), dependencies (Dependencies.cmake)
engine/include/   Public headers: engine/core/*, engine/voxel/*
engine/src/       Engine implementation (static library)
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
