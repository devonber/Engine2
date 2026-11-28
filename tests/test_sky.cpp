#include "engine/render/Sky.hpp"

#include <doctest/doctest.h>

#include <cmath>

using namespace engine;
using namespace engine::render;

TEST_CASE("Sun path: noon highest, below the horizon at night, rises east and sets west")
{
    CHECK(sunDirectionAt(12.0f).y > 0.85f);
    CHECK(sunDirectionAt(0.0f).y < -0.85f);
    CHECK(sunDirectionAt(9.0f).x < 0.0f);  // morning: east (-X)
    CHECK(sunDirectionAt(15.0f).x > 0.0f); // afternoon: west (+X)
    for (f32 h = 0.0f; h < 24.0f; h += 0.5f) {
        CHECK(std::abs(glm::length(sunDirectionAt(h)) - 1.0f) < 1e-5f);
    }
}

TEST_CASE("Sky light: bright by day, dark by night, moon opposite the sun, and continuous over the day")
{
    const SkyLight noon = skyLightFor(sunDirectionAt(12.0f));
    const SkyLight night = skyLightFor(sunDirectionAt(0.0f));
    CHECK(noon.lightColor.r > 0.5f);
    CHECK(noon.lightDirection.y > 0.0f);
    CHECK(night.lightDirection.y > 0.0f); // the moon is up when the sun is down
    CHECK(glm::dot(night.lightColor, glm::vec3{1.0f}) < 0.5f);
    CHECK(glm::dot(noon.skyColor, glm::vec3{1.0f}) > 5.0f * glm::dot(night.skyColor, glm::vec3{1.0f}));

    // No jumps from minute to minute (e.g. when the sun hands over to the moon).
    const auto brightness = [](const SkyLight& s) {
        return glm::dot(s.lightColor + s.ambientSky + s.skyColor, glm::vec3{1.0f});
    };
    SkyLight previous = skyLightFor(sunDirectionAt(0.0f));
    for (f32 h = 1.0f / 60.0f; h < 24.0f; h += 1.0f / 60.0f) {
        CAPTURE(h);
        const SkyLight s = skyLightFor(sunDirectionAt(h));
        CHECK(std::abs(brightness(s) - brightness(previous)) < 0.1f); // sunrise is quick, but smooth
        previous = s;
    }
}
