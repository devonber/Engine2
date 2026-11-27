#pragma once

// Profiling hooks. With ENGINE_ENABLE_TRACY=ON (the *-profile presets) these
// become Tracy zones. Otherwise they compile to nothing.

#if defined(TRACY_ENABLE)
    #include <tracy/Tracy.hpp>
    #define ENGINE_PROFILE_SCOPE(name) ZoneScopedN(name)
    #define ENGINE_PROFILE_FUNCTION() ZoneScoped
    #define ENGINE_PROFILE_FRAME() FrameMark
#else
    #define ENGINE_PROFILE_SCOPE(name) static_cast<void>(0)
    #define ENGINE_PROFILE_FUNCTION() static_cast<void>(0)
    #define ENGINE_PROFILE_FRAME() static_cast<void>(0)
#endif
