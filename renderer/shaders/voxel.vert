#version 460

// Vertex pulling: there are no vertex buffers. Each visible block face is one
// 8-byte PackedQuad (see engine/voxel/Mesher.hpp) in a storage buffer, and
// the shader expands it into 4 corners. A shared index buffer stitches the
// corners into 2 triangles (0-1-2, 2-1-3), so each corner is shaded once.
//
// Draws come from vkCmdDrawIndexedIndirect, one command per visible chunk:
//   vertexOffset  = firstQuad * 4  -> gl_VertexIndex / 4 is the global quad index
//   firstInstance = chunk index    -> gl_InstanceIndex selects the chunk origin

layout(std430, set = 0, binding = 0) readonly buffer QuadBuffer { uvec2 quads[]; };
layout(std430, set = 0, binding = 1) readonly buffer ChunkBuffer { ivec4 chunkOrigins[]; };

layout(push_constant) uniform PushConstants {
    mat4 viewProj;
    vec4 cameraPos; // xyz
    vec4 fog;       // rgb = colour (linear), a = distance of full fog
} pc;

layout(location = 0) flat out vec3 outColor;
layout(location = 1) out float outDistance;

// Per face (PosX, NegX, PosY, NegY, PosZ, NegZ): base corner, U axis, V axis.
// U x V points out of the block, so corners (0,0) (1,0) (0,1) (1,1) wound
// 0-1-2 / 2-1-3 are counter-clockwise seen from outside.
const vec3 kBase[6] = vec3[6](vec3(1, 0, 0), vec3(0, 0, 0), vec3(0, 1, 0), vec3(0, 0, 0), vec3(0, 0, 1), vec3(0, 0, 0));
const vec3 kU[6]    = vec3[6](vec3(0, 1, 0), vec3(0, 0, 1), vec3(0, 0, 1), vec3(1, 0, 0), vec3(1, 0, 0), vec3(0, 1, 0));
const vec3 kV[6]    = vec3[6](vec3(0, 0, 1), vec3(0, 1, 0), vec3(1, 0, 0), vec3(0, 0, 1), vec3(0, 1, 0), vec3(1, 0, 0));

// Cheap directional lighting baked per face.
const float kShade[6] = float[6](0.80, 0.80, 1.00, 0.50, 0.65, 0.65);

// Linear-space colours by block id: 0 air (magenta = bug), 1 stone, 2 dirt, 3 grass.
const vec3 kBlockColor[4] = vec3[4](vec3(1.0, 0.0, 1.0), vec3(0.20, 0.20, 0.22), vec3(0.16, 0.08, 0.035),
                                    vec3(0.10, 0.28, 0.045));

uint hash3(uvec3 p)
{
    uint h = (p.x * 73856093u) ^ (p.y * 19349663u) ^ (p.z * 83492791u);
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;
    return h;
}

void main()
{
    const uint quadIndex = uint(gl_VertexIndex) >> 2;
    const uint corner = uint(gl_VertexIndex) & 3u;

    const uvec2 q = quads[quadIndex];
    const uvec3 local = uvec3(q.x & 63u, (q.x >> 6) & 63u, (q.x >> 12) & 63u);
    const uint face = (q.x >> 18) & 7u;
    const uint block = q.y & 0xFFFFu;

    const ivec3 voxel = chunkOrigins[gl_InstanceIndex].xyz + ivec3(local);
    const vec2 uv = vec2(float(corner & 1u), float(corner >> 1));
    const vec3 pos = vec3(voxel) + kBase[face] + kU[face] * uv.x + kV[face] * uv.y;
    gl_Position = pc.viewProj * vec4(pos, 1.0);

    // Per-voxel brightness jitter so flat areas read as individual blocks.
    const float jitter = 0.92 + 0.16 * float(hash3(uvec3(voxel)) & 255u) / 255.0;
    outColor = kBlockColor[min(block, 3u)] * kShade[face] * jitter;
    outDistance = distance(pos, pc.cameraPos.xyz);
}
