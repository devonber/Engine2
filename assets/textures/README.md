# Block textures

PNG files named after the texture names in `engine/include/engine/voxel/Blocks.hpp`:

| File             | Used by                       |
|------------------|-------------------------------|
| `stone.png`      | stone (all faces)             |
| `dirt.png`       | dirt (all faces), grass bottom |
| `grass_top.png`  | grass top                     |
| `grass_side.png` | grass sides                   |
| `sand.png`       | sand (all faces)              |

All textures must be the same square, power-of-two size (e.g. 32x32). Side textures are drawn
upright: the top row of the image is at the top of the block. A missing file shows as a
magenta/black checkerboard and prints a warning.

The sandbox reads this folder directly, so edited textures appear on the next run without a rebuild.
