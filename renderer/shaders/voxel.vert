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
layout(std430, set = 0, binding = 3) readonly buffer FaceLayerBuffer { uint faceLayers[]; }; // [block * 6 + face]

layout(push_constant) uniform PushConstants {
    mat4 viewProj;
    vec4 cameraPos; // xyz
    vec4 fog;       // rgb = colour (linear), a = distance of full fog
} pc;

layout(location = 0) out vec2 outUV;
layout(location = 1) out float outDistance;
layout(location = 2) flat out uint outLayer;
layout(location = 3) flat out float outShade;

// Per face (PosX, NegX, PosY, NegY, PosZ, NegZ): base corner, U axis, V axis.
// Must match kFaceAxes in engine/voxel/Mesher.hpp (width runs along U, height along V).
// U x V points out of the block, so corners (0,0) (1,0) (0,1) (1,1) wound
// 0-1-2 / 2-1-3 are counter-clockwise seen from outside.
const vec3 kBase[6] = vec3[6](vec3(1, 0, 0), vec3(0, 0, 0), vec3(0, 1, 0), vec3(0, 0, 0), vec3(0, 0, 1), vec3(0, 0, 0));
const vec3 kU[6]    = vec3[6](vec3(0, 1, 0), vec3(0, 0, 1), vec3(0, 0, 1), vec3(1, 0, 0), vec3(1, 0, 0), vec3(0, 1, 0));
const vec3 kV[6]    = vec3[6](vec3(0, 0, 1), vec3(0, 1, 0), vec3(1, 0, 0), vec3(0, 0, 1), vec3(0, 1, 0), vec3(1, 0, 0));

// Cheap directional lighting baked per face.
const float kShade[6] = float[6](0.80, 0.80, 1.00, 0.50, 0.65, 0.65);

// Texture coordinates from the corner's position relative to the quad's first
// block (0..width / 0..height). The sampler repeats, so a merged quad shows one
// texture tile per block, exactly like separate faces. Side faces put the texture's top edge at the block's top and
// read left-to-right when seen from outside; top/bottom faces map x/z.
vec2 faceUV(uint face, vec3 o)
{
    switch (face) {
    case 0u: return vec2(1.0 - o.z, 1.0 - o.y); // +X
    case 1u: return vec2(o.z, 1.0 - o.y);       // -X
    case 2u: return vec2(o.x, o.z);             // +Y
    case 3u: return vec2(o.x, 1.0 - o.z);       // -Y
    case 4u: return vec2(o.x, 1.0 - o.y);       // +Z
    default: return vec2(1.0 - o.x, 1.0 - o.y); // -Z
    }
}

void main()
{
    const uint quadIndex = uint(gl_VertexIndex) >> 2;
    const uint corner = uint(gl_VertexIndex) & 3u;

    const uvec2 q = quads[quadIndex];
    const uvec3 local = uvec3(q.x & 63u, (q.x >> 6) & 63u, (q.x >> 12) & 63u);
    const uint face = (q.x >> 18) & 7u;
    const uint block = q.y & 0xFFFFu;
    // Greedy-merged quads cover width x height block faces.
    const vec2 size = vec2(float(((q.x >> 21) & 31u) + 1u), float(((q.x >> 26) & 31u) + 1u));

    const ivec3 voxel = chunkOrigins[gl_InstanceIndex].xyz + ivec3(local);
    const vec2 corner2 = vec2(float(corner & 1u), float(corner >> 1));
    const vec3 inBlock = kBase[face] + kU[face] * (corner2.x * size.x) + kV[face] * (corner2.y * size.y);
    const vec3 pos = vec3(voxel) + inBlock;
    gl_Position = pc.viewProj * vec4(pos, 1.0);

    outUV = faceUV(face, inBlock);
    outLayer = faceLayers[block * 6u + face];
    outShade = kShade[face];
    outDistance = distance(pos, pc.cameraPos.xyz);
}
