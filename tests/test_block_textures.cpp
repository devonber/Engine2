#include "engine/render/BlockTextures.hpp"
#include "engine/voxel/Mesher.hpp"

#include <doctest/doctest.h>

using namespace engine;
using namespace engine::render;
using namespace engine::voxel;

TEST_CASE("generateMips averages in linear space")
{
    // 2x2: two white, two black texels. A naive sRGB average gives 128;
    // the correct (linear) average of 0.5 encodes to sRGB 188.
    TextureArrayData data;
    data.size = 2;
    data.layers = 1;
    data.pixels = {255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 255};
    generateMips(data);

    REQUIRE(data.mipLevels == 2);
    REQUIRE(data.levelOffsets == std::vector<usize>{0, 16});
    REQUIRE(data.pixels.size() == 20);
    const u8* mip1 = data.pixels.data() + data.levelOffsets[1];
    CHECK(mip1[0] == 188);
    CHECK(mip1[1] == 188);
    CHECK(mip1[2] == 188);
    CHECK(mip1[3] == 255);
}

TEST_CASE("Block faces map to texture layers")
{
    // Nonexistent folder: every texture falls back to a placeholder, but the
    // layer assignment and mip chain must still be complete.
    const BlockTextureSet set = loadBlockTextures(blockRegistry(), "does/not/exist");

    CHECK(set.layerNames.size() == 5);
    CHECK(set.textures.layers == 5);
    CHECK(set.textures.size == 32);
    CHECK(set.textures.mipLevels == 6); // 32, 16, 8, 4, 2, 1
    REQUIRE(set.faceLayers.size() == blockRegistry().size() * 6);

    const auto layer = [&](BlockId block, Face face) {
        return set.layerNames[set.faceLayers[block * 6u + static_cast<u32>(face)]];
    };
    CHECK(layer(blocks::Grass, Face::PosY) == "grass_top");
    CHECK(layer(blocks::Grass, Face::NegY) == "dirt");
    CHECK(layer(blocks::Grass, Face::PosX) == "grass_side");
    CHECK(layer(blocks::Grass, Face::NegZ) == "grass_side");
    CHECK(layer(blocks::Dirt, Face::PosY) == "dirt");
    CHECK(layer(blocks::Sand, Face::NegX) == "sand");
    CHECK(layer(blocks::Stone, Face::PosZ) == "stone");
}
