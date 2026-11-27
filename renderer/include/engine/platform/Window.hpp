#pragma once

#include "engine/core/Types.hpp"
#include "engine/render/Vk.hpp"

#include <span>

struct GLFWwindow;

namespace engine::platform {

enum class Key { W, A, S, D, Space, LeftShift, LeftControl, Escape, Count };

struct InputState {
    bool keys[static_cast<usize>(Key::Count)] = {};
    f64 mouseDx = 0.0; // accumulated since the last pollEvents()
    f64 mouseDy = 0.0;
    bool mouseCaptured = false;

    [[nodiscard]] bool down(Key k) const noexcept { return keys[static_cast<usize>(k)]; }
};

struct Extent {
    u32 width = 0;
    u32 height = 0;
};

// Thin GLFW wrapper: one window, Vulkan surface creation, keyboard/mouse.
class Window {
public:
    Window(const char* title, u32 width, u32 height);
    ~Window();
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    void pollEvents();
    [[nodiscard]] bool shouldClose() const;
    void requestClose();
    void setTitle(const char* title);

    // Size of the drawable area in pixels (0x0 while minimised).
    [[nodiscard]] Extent framebufferExtent() const;
    void waitWhileMinimised();

    // True once after the framebuffer changed size.
    [[nodiscard]] bool consumeResized() noexcept;

    [[nodiscard]] const InputState& input() const noexcept { return m_input; }

    [[nodiscard]] static std::span<const char* const> requiredInstanceExtensions();
    [[nodiscard]] VkSurfaceKHR createSurface(VkInstance instance) const;

private:
    static void onFramebufferResize(GLFWwindow* window, int width, int height);
    static void onMouseButton(GLFWwindow* window, int button, int action, int mods);
    void setMouseCaptured(bool captured);

    GLFWwindow* m_window = nullptr;
    InputState m_input;
    f64 m_lastMouseX = 0.0;
    f64 m_lastMouseY = 0.0;
    bool m_resized = false;
};

} // namespace engine::platform
