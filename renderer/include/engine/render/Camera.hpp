#pragma once

#include "engine/core/Types.hpp"

#include <array>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace engine::render {

struct Camera {
    glm::vec3 position{0.0f, 40.0f, 0.0f};
    f32 yaw = 0.0f;   // radians; 0 looks down -Z
    f32 pitch = 0.0f; // radians; positive looks up
    f32 fovY = glm::radians(70.0f);
    f32 nearPlane = 0.05f;

    [[nodiscard]] glm::vec3 forward() const noexcept
    {
        const f32 cp = std::cos(pitch);
        return {std::sin(yaw) * cp, std::sin(pitch), -std::cos(yaw) * cp};
    }

    [[nodiscard]] glm::vec3 right() const noexcept { return glm::normalize(glm::cross(forward(), {0.0f, 1.0f, 0.0f})); }

    [[nodiscard]] glm::mat4 view() const noexcept
    {
        return glm::lookAtRH(position, position + forward(), glm::vec3{0.0f, 1.0f, 0.0f});
    }

    // Reversed-Z, infinite far plane, Vulkan clip space (Y down, depth 0..1).
    // Depth is 1 at the near plane and approaches 0 at infinity. With a float
    // depth buffer this spreads precision almost evenly over distance, so far
    // terrain does not z-fight, and there is no far plane to clip against.
    [[nodiscard]] glm::mat4 projection(f32 aspect) const noexcept
    {
        const f32 f = 1.0f / std::tan(fovY * 0.5f);
        glm::mat4 m{0.0f};
        m[0][0] = f / aspect;
        m[1][1] = -f; // flip Y for Vulkan
        m[2][3] = -1.0f;
        m[3][2] = nearPlane;
        return m;
    }
};

// View frustum as 5 planes (no far plane: the projection is infinite).
// A point p is inside a plane when dot(plane.xyz, p) + plane.w >= 0.
struct Frustum {
    std::array<glm::vec4, 5> planes;

    // Gribb-Hartmann plane extraction from a view-projection matrix.
    [[nodiscard]] static Frustum fromViewProj(const glm::mat4& m) noexcept
    {
        const auto row = [&](int r) { return glm::vec4{m[0][r], m[1][r], m[2][r], m[3][r]}; };
        const glm::vec4 r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);
        Frustum fr{{r3 + r0, r3 - r0, r3 + r1, r3 - r1, r3 - r2}}; // left right bottom top near
        for (auto& p : fr.planes) {
            p /= glm::length(glm::vec3{p});
        }
        return fr;
    }

    // Conservative AABB test: false only if the box is fully outside a plane.
    [[nodiscard]] bool intersects(const glm::vec3& min, const glm::vec3& max) const noexcept
    {
        for (const auto& p : planes) {
            // The box corner furthest along the plane normal.
            const glm::vec3 v{p.x >= 0.0f ? max.x : min.x, p.y >= 0.0f ? max.y : min.y, p.z >= 0.0f ? max.z : min.z};
            if (glm::dot(glm::vec3{p}, v) + p.w < 0.0f) {
                return false;
            }
        }
        return true;
    }
};

} // namespace engine::render
