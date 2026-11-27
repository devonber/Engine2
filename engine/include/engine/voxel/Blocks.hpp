#pragma once

#include "engine/voxel/Chunk.hpp"

#include <span>

namespace engine::voxel {

namespace blocks {
inline constexpr BlockId Air = kAir;
inline constexpr BlockId Stone = 1;
inline constexpr BlockId Dirt = 2;
inline constexpr BlockId Grass = 3;
inline constexpr BlockId Sand = 4;
} // namespace blocks

// Static description of a block type. Texture names refer to
// assets/textures/<name>.png. Index into the registry = BlockId.
struct BlockInfo {
    const char* name;
    const char* textureTop;
    const char* textureBottom;
    const char* textureSide;
};

inline constexpr BlockInfo kBlockRegistry[] = {
    {"air", nullptr, nullptr, nullptr}, {"stone", "stone", "stone", "stone"},
    {"dirt", "dirt", "dirt", "dirt"},   {"grass", "grass_top", "dirt", "grass_side"},
    {"sand", "sand", "sand", "sand"},
};

[[nodiscard]] constexpr std::span<const BlockInfo> blockRegistry() noexcept
{
    return kBlockRegistry;
}

} // namespace engine::voxel
