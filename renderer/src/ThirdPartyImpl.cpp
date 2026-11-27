// Implementation units for header-only libraries. Compiled without our
// strict warning flags (see renderer/CMakeLists.txt).
#define VOLK_IMPLEMENTATION
#include <volk.h>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_FAILURE_USERMSG
#include <stb_image.h>
