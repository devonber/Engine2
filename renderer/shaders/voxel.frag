#version 460

layout(push_constant) uniform PushConstants {
    mat4 viewProj;
    vec4 cameraPos;
    vec4 fog; // rgb = colour (linear), a = distance of full fog
} pc;

layout(set = 0, binding = 2) uniform sampler2DArray blockTextures;

layout(location = 0) in vec2 inUV;
layout(location = 1) in float inDistance;
layout(location = 2) flat in uint inLayer;
layout(location = 3) flat in float inShade;
layout(location = 4) in float inOcclusion; // interpolated across the quad

layout(location = 0) out vec4 outColor;

void main()
{
    // inUV runs 0..width / 0..height across a greedy-merged quad: one texture
    // tile per block. Wrap it per block ourselves and sample with CLAMP_TO_EDGE
    // rather than relying on REPEAT: with REPEAT, filtering near a block's
    // edge (minification, mips, anisotropy) blends in the texels from the
    // OPPOSITE edge of the texture, e.g. a green line from the top of the
    // grass side texture along the bottom of every grass block.
    // Derivatives come from the unwrapped coordinates, so the jump at each
    // block boundary does not make the GPU pick the smallest mip there.
    const vec2 tile = fract(inUV);
    const vec3 albedo =
        textureGrad(blockTextures, vec3(tile, float(inLayer)), dFdx(inUV), dFdy(inUV)).rgb;

    // Fog hides the edge of the loaded world and matches the sky clear colour.
    const float f = smoothstep(pc.fog.a * 0.55, pc.fog.a, inDistance);
    outColor = vec4(mix(albedo * (inShade * inOcclusion), pc.fog.rgb, f), 1.0);
}
