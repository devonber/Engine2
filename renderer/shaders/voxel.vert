#version 460

// Vertex pulling: there are no vertex buffers. Each visible block face is one
// 8-byte PackedQuad (see engine/voxel/Mesher.hpp) in a storage buffer, and
// the shader expands it into 4 corners. A shared index buffer stitches the
// corners into 2 triangles (0-1-2, 2-1-3), so each corner is shaded once.
//
// Draws come from vkCmdDrawIndexedIndirect, one command per visible chunk:
//   vertexOffset  = firstQuad * 4  -> gl_VertexIndex / 4 is the global quad index
//   firstInstance = chunk slot     -> gl_InstanceIndex selects the chunk origin

layout(std430, set = 0, binding = 0) readonly buffer QuadBuffer { uvec2 quads[]; };
// Per chunk slot (see GpuChunk in renderer/src/ChunkMeshStore.hpp); only the
// origin is needed here.
struct GpuChunk {
    ivec4 originFirst; // xyz: world position of voxel (0,0,0)
    uvec4 faces0;
    uvec4 faces1;
};
layout(std430, set = 0, binding = 1) readonly buffer ChunkTable { GpuChunk chunks[]; };
layout(std430, set = 0, binding = 3) readonly buffer FaceLayerBuffer { uint faceLayers[]; }; // [block * 6 + face]

layout(push_constant) uniform PushConstants {
    mat4 viewProj;       // projection * rotation: the camera sits at the origin
    ivec4 cameraBlock;   // xyz: integer block containing the camera
    vec4 cameraFraction; // xyz: camera position within that block
    vec4 fog;            // rgb = colour (linear), a = distance of full fog
} pc;

layout(location = 0) out vec2 outUV;
layout(location = 1) out float outDistance;
layout(location = 2) flat out uint outLayer;
layout(location = 3) flat out float outShade;
layout(location = 4) out float outOcclusion;

// Per face (PosX, NegX, PosY, NegY, PosZ, NegZ): base corner, U axis, V axis.
// Must match kFaceAxes in engine/voxel/Mesher.hpp (width runs along U, height along V).
// U x V points out of the block, so corners (0,0) (1,0) (0,1) (1,1) wound
// 0-1-2 / 2-1-3 are counter-clockwise seen from outside.
const vec3 kBase[6] = vec3[6](vec3(1, 0, 0), vec3(0, 0, 0), vec3(0, 1, 0), vec3(0, 0, 0), vec3(0, 0, 1), vec3(0, 0, 0));
const vec3 kU[6]    = vec3[6](vec3(0, 1, 0), vec3(0, 0, 1), vec3(0, 0, 1), vec3(1, 0, 0), vec3(1, 0, 0), vec3(0, 1, 0));
const vec3 kV[6]    = vec3[6](vec3(0, 0, 1), vec3(0, 1, 0), vec3(1, 0, 0), vec3(0, 0, 1), vec3(0, 1, 0), vec3(1, 0, 0));

// Cheap directional lighting baked per face.
const float kShade[6] = float[6](0.80, 0.80, 1.00, 0.50, 0.65, 0.65);

// Brightness per ambient-occlusion level (0 = enclosed corner .. 3 = open).
const float kOcclusion[4] = float[4](0.42, 0.62, 0.81, 1.0);

// Vertex slot -> quad corner when the quad's diagonal is flipped. The index
// buffer always makes triangles (0,1,2) (2,1,3), splitting along corners 1-2;
// with this mapping they become (1,3,0) (0,3,2), split along 0-3, and still
// counter-clockwise.
const uint kFlipped[4] = uint[4](1u, 3u, 0u, 2u);

// Texture coordinates from the corner's position relative to the quad's first
// block (0..width / 0..height). The fragment shader wraps it per block, so a merged quad shows one
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
    const uint slot = uint(gl_VertexIndex) & 3u;

    const uvec2 q = quads[quadIndex];

    // Ambient occlusion: 2 bits per corner (corner = cu + 2 * cv). Interpolating
    // across a quad depends on which diagonal splits it; split along the pair
    // of corners that are together less occluded, so a single dark corner
    // fades out towards its neighbours instead of streaking along the diagonal.
    const uint aoBits = (q.y >> 16) & 0xFFu;
    const uint ao0 = aoBits & 3u, ao1 = (aoBits >> 2) & 3u, ao2 = (aoBits >> 4) & 3u, ao3 = (aoBits >> 6) & 3u;
    const uint corner = (ao0 + ao3 > ao1 + ao2) ? kFlipped[slot] : slot;
    const uvec3 local = uvec3(q.x & 63u, (q.x >> 6) & 63u, (q.x >> 12) & 63u);
    const uint face = (q.x >> 18) & 7u;
    const uint block = q.y & 0xFFFFu;
    // Greedy-merged quads cover width x height block faces.
    const vec2 size = vec2(float(((q.x >> 21) & 31u) + 1u), float(((q.x >> 26) & 31u) + 1u));

    // Camera-relative: the integer part is exact, so precision does not
    // depend on how far from the world origin the camera is.
    const ivec3 voxel = chunks[gl_InstanceIndex].originFirst.xyz + ivec3(local) - pc.cameraBlock.xyz;
    const vec2 corner2 = vec2(float(corner & 1u), float(corner >> 1));
    const vec3 inBlock = kBase[face] + kU[face] * (corner2.x * size.x) + kV[face] * (corner2.y * size.y);
    const vec3 pos = vec3(voxel) - pc.cameraFraction.xyz + inBlock; // relative to the camera
    gl_Position = pc.viewProj * vec4(pos, 1.0);

    outUV = faceUV(face, inBlock);
    outLayer = faceLayers[block * 6u + face];
    outShade = kShade[face];
    outOcclusion = kOcclusion[(aoBits >> (2u * corner)) & 3u];
    outDistance = length(pos);
}
