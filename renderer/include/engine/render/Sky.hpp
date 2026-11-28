#pragma once

#include "engine/core/Types.hpp"

#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>

namespace engine::render {

// Direction towards the sun at a time of day (hours, 0..24; 12 = noon). The
// sun rises in the east (-X), is highest in the south (+Z side) at noon and
// sets in the west (+X); it is below the horizon from about 18:00 to 6:00.
[[nodiscard]] inline glm::vec3 sunDirectionAt(f32 hours) noexcept
{
    const f32 angle = (hours - 12.0f) / 24.0f * 2.0f * 3.14159265f; // 0 at noon
    return glm::normalize(glm::vec3{std::sin(angle), std::cos(angle) * 0.9f, 0.45f});
}

// The light of the scene for a sun direction, in linear RGB. Kept simple and
// cheap: no scattering simulation, just smooth blends between day, sunset
// and night that look plausible.
struct SkyLight {
    glm::vec3 lightDirection{0.0f, 1.0f, 0.0f}; // towards the sun, or the moon at night
    glm::vec3 lightColor{1.0f};                 // direct light (sun or moon)
    glm::vec3 ambientSky{0.4f};                 // light from the sky hemisphere, onto up-facing faces
    glm::vec3 ambientGround{0.2f};              // light bounced from the ground, onto down-facing faces
    glm::vec3 skyColor{0.36f, 0.58f, 0.92f};    // clear colour, and fog: distant terrain fades into it
};

[[nodiscard]] inline SkyLight skyLightFor(glm::vec3 towardSun) noexcept
{
    const auto smooth = [](f32 a, f32 b, f32 x) {
        const f32 t = std::clamp((x - a) / (b - a), 0.0f, 1.0f);
        return t * t * (3.0f - 2.0f * t);
    };
    towardSun = glm::normalize(towardSun);
    const f32 elevation = towardSun.y; // sine of the sun's elevation

    const f32 day = smooth(-0.15f, 0.2f, elevation);                            // 0 night .. 1 day
    const f32 sunset = 1.0f - smooth(0.0f, 0.35f, std::abs(elevation + 0.02f)); // near the horizon
    const glm::vec3 daySky{0.36f, 0.58f, 0.92f};
    const glm::vec3 nightSky{0.012f, 0.018f, 0.045f};
    const glm::vec3 sunsetSky{1.0f, 0.5f, 0.25f};

    SkyLight s;
    s.skyColor = glm::mix(glm::mix(nightSky, daySky, day), sunsetSky, 0.7f * sunset * smooth(-0.2f, 0.0f, elevation));

    if (elevation > -0.05f) {
        // Sun: warm and dim near the horizon, white at height; fades out as it sets.
        const glm::vec3 low{1.0f, 0.5f, 0.25f};
        const glm::vec3 high{1.0f, 0.96f, 0.9f};
        s.lightDirection = towardSun;
        s.lightColor = glm::mix(low, high, smooth(0.0f, 0.45f, elevation)) * 0.8f * smooth(-0.05f, 0.12f, elevation);
    } else {
        // Moon, opposite the sun: faint and blue.
        s.lightDirection = -towardSun;
        s.lightColor = glm::vec3{0.10f, 0.12f, 0.20f} * smooth(0.05f, 0.25f, -elevation);
    }
    s.ambientSky = glm::mix(glm::vec3{0.025f, 0.03f, 0.06f}, glm::vec3{0.38f, 0.42f, 0.50f}, day) +
                   glm::vec3{0.10f, 0.05f, 0.02f} * sunset * day;
    s.ambientGround = glm::mix(glm::vec3{0.01f, 0.012f, 0.02f}, glm::vec3{0.20f, 0.19f, 0.16f}, day);
    return s;
}

} // namespace engine::render
