#include "engine/render/BlockTextures.hpp"

#include "engine/core/Profile.hpp"
#include "engine/voxel/Mesher.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stb_image.h>
#include <unordered_map>

namespace engine::render {

namespace {

constexpr u32 kFallbackSize = 32;

struct Image {
    u32 size = 0;
    std::vector<u8> rgba;
};

Image checkerboard(u32 size)
{
    Image img{size, std::vector<u8>(static_cast<usize>(size) * size * 4)};
    for (u32 y = 0; y < size; ++y) {
        for (u32 x = 0; x < size; ++x) {
            const bool magenta = ((x / 8) + (y / 8)) % 2 == 0;
            u8* p = &img.rgba[(static_cast<usize>(y) * size + x) * 4];
            p[0] = magenta ? 255 : 0;
            p[1] = 0;
            p[2] = magenta ? 255 : 0;
            p[3] = 255;
        }
    }
    return img;
}

// Returns an empty image (size 0) on failure, after printing why.
Image loadPng(const std::filesystem::path& path)
{
    int w = 0, h = 0, channels = 0;
    const std::string pathStr = path.string();
    stbi_uc* data = stbi_load(pathStr.c_str(), &w, &h, &channels, 4);
    if (data == nullptr) {
        std::fprintf(stderr, "Texture '%s': %s\n", pathStr.c_str(), stbi_failure_reason());
        return {};
    }
    Image img;
    if (w != h || !std::has_single_bit(static_cast<u32>(w))) {
        std::fprintf(stderr, "Texture '%s' is %dx%d; textures must be square with a power-of-two size.\n",
                     pathStr.c_str(), w, h);
    } else {
        img.size = static_cast<u32>(w);
        img.rgba.assign(data, data + static_cast<usize>(w) * h * 4);
    }
    stbi_image_free(data);
    return img;
}

const std::array<f32, 256>& srgbToLinearTable()
{
    static const std::array<f32, 256> table = [] {
        std::array<f32, 256> t{};
        for (usize i = 0; i < 256; ++i) {
            const f32 c = static_cast<f32>(i) / 255.0f;
            t[i] = c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
        }
        return t;
    }();
    return table;
}

u8 linearToSrgb8(f32 c)
{
    c = std::clamp(c, 0.0f, 1.0f);
    const f32 s = c <= 0.0031308f ? c * 12.92f : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
    return static_cast<u8>(std::lround(s * 255.0f));
}

} // namespace

void generateMips(TextureArrayData& data)
{
    ENGINE_PROFILE_FUNCTION();
    const auto& toLinear = srgbToLinearTable();
    data.mipLevels = static_cast<u32>(std::bit_width(data.size)); // 32 -> 6 levels (32..1)
    data.levelOffsets.resize(data.mipLevels);

    usize total = 0;
    for (u32 level = 0; level < data.mipLevels; ++level) {
        data.levelOffsets[level] = total;
        const usize s = data.levelSize(level);
        total += s * s * 4 * data.layers;
    }
    data.pixels.resize(total);

    for (u32 level = 1; level < data.mipLevels; ++level) {
        const u32 src = data.levelSize(level - 1);
        const u32 dst = data.levelSize(level);
        for (u32 layer = 0; layer < data.layers; ++layer) {
            const u8* in =
                data.pixels.data() + data.levelOffsets[level - 1] + static_cast<usize>(layer) * src * src * 4;
            u8* out = data.pixels.data() + data.levelOffsets[level] + static_cast<usize>(layer) * dst * dst * 4;
            for (u32 y = 0; y < dst; ++y) {
                for (u32 x = 0; x < dst; ++x) {
                    const u8* p[4] = {
                        in + ((2 * y) * src + 2 * x) * 4,
                        in + ((2 * y) * src + 2 * x + 1) * 4,
                        in + ((2 * y + 1) * src + 2 * x) * 4,
                        in + ((2 * y + 1) * src + 2 * x + 1) * 4,
                    };
                    u8* o = out + (y * dst + x) * 4;
                    for (int c = 0; c < 3; ++c) {
                        const f32 sum = toLinear[p[0][c]] + toLinear[p[1][c]] + toLinear[p[2][c]] + toLinear[p[3][c]];
                        o[c] = linearToSrgb8(sum * 0.25f);
                    }
                    o[3] = static_cast<u8>((p[0][3] + p[1][3] + p[2][3] + p[3][3] + 2) / 4); // alpha is linear
                }
            }
        }
    }
}

BlockTextureSet loadBlockTextures(std::span<const voxel::BlockInfo> registry, const std::filesystem::path& textureDir)
{
    ENGINE_PROFILE_FUNCTION();
    BlockTextureSet set;

    // Assign one array layer per distinct texture name, in first-use order.
    std::unordered_map<std::string, u32> layerOf;
    const auto layerFor = [&](const char* name) -> u32 {
        if (name == nullptr) {
            return 0;
        }
        const auto [it, inserted] = layerOf.try_emplace(name, static_cast<u32>(set.layerNames.size()));
        if (inserted) {
            set.layerNames.emplace_back(name);
        }
        return it->second;
    };

    set.faceLayers.resize(registry.size() * 6);
    for (usize block = 0; block < registry.size(); ++block) {
        const voxel::BlockInfo& info = registry[block];
        const u32 top = layerFor(info.textureTop);
        const u32 bottom = layerFor(info.textureBottom);
        const u32 side = layerFor(info.textureSide);
        u32* faces = &set.faceLayers[block * 6];
        faces[static_cast<usize>(voxel::Face::PosX)] = side;
        faces[static_cast<usize>(voxel::Face::NegX)] = side;
        faces[static_cast<usize>(voxel::Face::PosY)] = top;
        faces[static_cast<usize>(voxel::Face::NegY)] = bottom;
        faces[static_cast<usize>(voxel::Face::PosZ)] = side;
        faces[static_cast<usize>(voxel::Face::NegZ)] = side;
    }
    if (set.layerNames.empty()) {
        set.layerNames.emplace_back("<none>");
    }

    // Load. The first valid texture decides the size; others must match it.
    std::vector<Image> images;
    images.reserve(set.layerNames.size());
    u32 size = 0;
    for (const std::string& name : set.layerNames) {
        Image img = loadPng(textureDir / (name + ".png"));
        if (img.size != 0 && size != 0 && img.size != size) {
            std::fprintf(stderr, "Texture '%s' is %ux%u but other block textures are %ux%u.\n", name.c_str(), img.size,
                         img.size, size, size);
            img = {};
        }
        if (img.size != 0 && size == 0) {
            size = img.size;
        }
        images.push_back(std::move(img));
    }
    if (size == 0) {
        size = kFallbackSize;
    }
    for (usize i = 0; i < images.size(); ++i) {
        if (images[i].size == 0) {
            std::fprintf(stderr, "  -> using a placeholder checkerboard for '%s'\n", set.layerNames[i].c_str());
            images[i] = checkerboard(size);
        }
    }

    TextureArrayData& tex = set.textures;
    tex.size = size;
    tex.layers = static_cast<u32>(images.size());
    tex.pixels.resize(static_cast<usize>(size) * size * 4 * tex.layers);
    for (usize i = 0; i < images.size(); ++i) {
        std::memcpy(tex.pixels.data() + i * size * size * 4, images[i].rgba.data(), images[i].rgba.size());
    }
    generateMips(tex);
    return set;
}

} // namespace engine::render
