#pragma once

#include "engine/core/Types.hpp"
#include "engine/voxel/Blocks.hpp"

#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace engine::render {

// RGBA8 (sRGB) 2D texture array with a full mip chain, laid out for upload:
// mip level 0 for every layer, then level 1 for every layer, and so on.
struct TextureArrayData {
    u32 size = 0; // width == height, power of two
    u32 layers = 0;
    u32 mipLevels = 0;
    std::vector<u8> pixels;
    std::vector<usize> levelOffsets; // byte offset of each mip level in `pixels`

    [[nodiscard]] u32 levelSize(u32 level) const noexcept { return size >> level; }
};

// Everything the GPU needs to texture blocks.
struct BlockTextureSet {
    TextureArrayData textures;
    std::vector<std::string> layerNames; // layer index -> texture name
    std::vector<u32> faceLayers;         // [blockId * 6 + face] -> layer
};

// Loads every texture referenced by the block registry from
// `textureDir/<name>.png`. All textures must be the same square, power-of-two
// size. A missing or invalid file is replaced by a magenta/black checkerboard
// (with a warning), so the game still runs.
[[nodiscard]] BlockTextureSet loadBlockTextures(std::span<const voxel::BlockInfo> registry,
                                                const std::filesystem::path& textureDir);

// Builds mip levels 1..n from level 0 of `data` (2x2 box filter in linear
// colour space, so mips do not darken). Exposed for tests.
void generateMips(TextureArrayData& data);

} // namespace engine::render
