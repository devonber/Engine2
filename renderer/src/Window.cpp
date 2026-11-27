#include "engine/platform/Window.hpp"

#include "engine/render/Vk.hpp"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <array>

namespace engine::platform {

namespace {

constexpr std::array<int, static_cast<usize>(Key::Count)> kGlfwKeys = {
    GLFW_KEY_W,      GLFW_KEY_A, GLFW_KEY_S, GLFW_KEY_D, GLFW_KEY_SPACE, GLFW_KEY_LEFT_SHIFT, GLFW_KEY_LEFT_CONTROL,
    GLFW_KEY_ESCAPE,
};

void glfwErrorCallback(int code, const char* description)
{
    std::fprintf(stderr, "GLFW error %d: %s\n", code, description);
}

} // namespace

Window::Window(const char* title, u32 width, u32 height)
{
    // Load the Vulkan loader (vulkan-1.dll / libvulkan.so.1) and its entry point.
    if (volkInitialize() != VK_SUCCESS) {
        render::detail::fatal("Vulkan loader not found. Install or update your GPU driver.");
    }
    glfwSetErrorCallback(glfwErrorCallback);
    // Make GLFW use the loader volk already opened instead of loading its own.
    glfwInitVulkanLoader(vkGetInstanceProcAddr);
    if (glfwInit() != GLFW_TRUE) {
        render::detail::fatal("glfwInit failed");
    }
    if (glfwVulkanSupported() != GLFW_TRUE) {
        render::detail::fatal("No Vulkan loader/driver found. Update your GPU driver.");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // Vulkan, not OpenGL
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    m_window = glfwCreateWindow(static_cast<int>(width), static_cast<int>(height), title, nullptr, nullptr);
    if (m_window == nullptr) {
        render::detail::fatal("glfwCreateWindow failed");
    }

    glfwSetWindowUserPointer(m_window, this);
    glfwSetFramebufferSizeCallback(m_window, onFramebufferResize);
    glfwSetMouseButtonCallback(m_window, onMouseButton);
    if (glfwRawMouseMotionSupported() == GLFW_TRUE) {
        glfwSetInputMode(m_window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    }
}

Window::~Window()
{
    glfwDestroyWindow(m_window);
    glfwTerminate();
}

void Window::pollEvents()
{
    glfwPollEvents();

    for (usize i = 0; i < kGlfwKeys.size(); ++i) {
        m_input.keys[i] = glfwGetKey(m_window, kGlfwKeys[i]) == GLFW_PRESS;
    }
    if (m_input.down(Key::Escape) && m_input.mouseCaptured) {
        setMouseCaptured(false);
    }

    f64 x = 0.0;
    f64 y = 0.0;
    glfwGetCursorPos(m_window, &x, &y);
    m_input.mouseDx = m_input.mouseCaptured ? x - m_lastMouseX : 0.0;
    m_input.mouseDy = m_input.mouseCaptured ? y - m_lastMouseY : 0.0;
    m_lastMouseX = x;
    m_lastMouseY = y;
}

bool Window::shouldClose() const
{
    return glfwWindowShouldClose(m_window) == GLFW_TRUE;
}

void Window::requestClose()
{
    glfwSetWindowShouldClose(m_window, GLFW_TRUE);
}

void Window::setTitle(const char* title)
{
    glfwSetWindowTitle(m_window, title);
}

Extent Window::framebufferExtent() const
{
    int w = 0;
    int h = 0;
    glfwGetFramebufferSize(m_window, &w, &h);
    return {static_cast<u32>(w), static_cast<u32>(h)};
}

void Window::waitWhileMinimised()
{
    Extent e = framebufferExtent();
    while ((e.width == 0 || e.height == 0) && !shouldClose()) {
        glfwWaitEvents();
        e = framebufferExtent();
    }
}

bool Window::consumeResized() noexcept
{
    const bool r = m_resized;
    m_resized = false;
    return r;
}

std::span<const char* const> Window::requiredInstanceExtensions()
{
    u32 count = 0;
    const char** names = glfwGetRequiredInstanceExtensions(&count);
    if (names == nullptr) {
        render::detail::fatal("GLFW cannot create Vulkan surfaces on this system");
    }
    return {names, count};
}

VkSurfaceKHR Window::createSurface(VkInstance instance) const
{
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VK_CHECK(glfwCreateWindowSurface(instance, m_window, nullptr, &surface));
    return surface;
}

void Window::onFramebufferResize(GLFWwindow* window, int /*width*/, int /*height*/)
{
    static_cast<Window*>(glfwGetWindowUserPointer(window))->m_resized = true;
}

void Window::onMouseButton(GLFWwindow* window, int button, int action, int /*mods*/)
{
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS && !self->m_input.mouseCaptured) {
        self->setMouseCaptured(true);
    }
}

void Window::setMouseCaptured(bool captured)
{
    m_input.mouseCaptured = captured;
    glfwSetInputMode(m_window, GLFW_CURSOR, captured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    glfwGetCursorPos(m_window, &m_lastMouseX, &m_lastMouseY); // no jump on capture
}

} // namespace engine::platform
