#version 460

layout(push_constant) uniform PushConstants {
    mat4 viewProj;
    ivec4 cameraBlock;
    vec4 cameraFraction;
    vec4 fog; // rgb = colour (linear, the sky), a = distance of full fog
} pc;

layout(set = 0, binding = 2) uniform sampler2DArray blockTextures;
layout(set = 0, binding = 4) uniform sampler2DArrayShadow shadowCascades;

// Per frame (see LightingUniforms in Renderer.cpp).
layout(set = 0, binding = 5) uniform Lighting {
    mat4 cascades[4];    // camera-relative position -> light clip space
    vec4 splits;         // view depth where each cascade ends
    vec4 texelSize;      // world size of a shadow texel, per cascade
    vec4 lightDirection; // xyz: towards the sun/moon; w: 1 if shadows are on
    vec4 lightColor;     // rgb; w: shadow distance
    vec4 ambientSky;
    vec4 ambientGround;
    vec4 cameraForward;
} lighting;

layout(location = 0) in vec2 inUV;
layout(location = 1) in float inDistance;
layout(location = 2) flat in uint inLayer;
layout(location = 3) flat in uint inFace;
layout(location = 4) in float inOcclusion; // interpolated across the quad
layout(location = 5) in vec3 inPos;        // camera-relative
layout(location = 6) flat in float inSeamRadius;

// True only in the pipeline used for coarse chunks that cross a level-of-detail
// seam: closer to the camera than the seam, the finer level is drawn instead.
// The test is per pixel and depends only on the horizontal distance from the
// camera, which only grows along a view ray, so every ray sees the fine level
// up to the seam and the coarse one after it.
layout(constant_id = 0) const bool kLodDiscard = false;

layout(location = 0) out vec4 outColor;

const vec3 kNormals[6] = vec3[6](vec3(1, 0, 0), vec3(-1, 0, 0), vec3(0, 1, 0), vec3(0, -1, 0), vec3(0, 0, 1), vec3(0, 0, -1));

// Fraction of direct light reaching this point (1 = lit), from the shadow
// cascades: 4 bilinear depth comparisons (16 texels, smoothly weighted).
float sunVisibility(vec3 normal, float viewDepth)
{
    const float maxDistance = lighting.lightColor.w;
    if (lighting.lightDirection.w == 0.0 || viewDepth >= maxDistance) {
        return 1.0;
    }
    int c = 0;
    while (c < 3 && viewDepth > lighting.splits[c]) {
        ++c;
    }
    // Normal offset: look up from slightly off the surface, by about a texel,
    // so a surface does not shadow itself.
    const float texel = lighting.texelSize[c];
    const vec3 p = inPos + normal * (texel * 1.5);
    const vec4 clip = lighting.cascades[c] * vec4(p, 1.0);
    const vec2 uv = clip.xy * 0.5 + 0.5;
    const float depth = clip.z;
    const vec2 step = vec2(1.0) / vec2(textureSize(shadowCascades, 0).xy);
    float lit = 0.0;
    for (int i = 0; i < 4; ++i) {
        const vec2 o = (vec2(float(i & 1), float(i >> 1)) - 0.5) * step;
        lit += texture(shadowCascades, vec4(uv + o, float(c), depth));
    }
    lit *= 0.25;
    // Fade out towards the shadow distance, so its edge is not visible.
    return mix(lit, 1.0, smoothstep(maxDistance * 0.8, maxDistance, viewDepth));
}

void main()
{
    if (kLodDiscard && dot(inPos.xz, inPos.xz) < inSeamRadius * inSeamRadius) {
        discard;
    }

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

    // Lighting: sky light from the hemisphere above (ground bounce from
    // below), darkened by ambient occlusion, plus direct sun (or moon) light,
    // shadowed. Occlusion also dims direct light a little: crevices stay
    // darker in sunlight too.
    const vec3 n = kNormals[inFace];
    const vec3 ambient = mix(lighting.ambientGround.rgb, lighting.ambientSky.rgb, n.y * 0.5 + 0.5);
    const float facing = max(dot(n, lighting.lightDirection.xyz), 0.0);
    float direct = 0.0;
    if (facing > 0.0) {
        direct = facing * sunVisibility(n, dot(inPos, lighting.cameraForward.xyz));
    }
    const vec3 light = ambient * inOcclusion + lighting.lightColor.rgb * (direct * mix(1.0, inOcclusion, 0.5));

    // Fog hides the edge of the loaded world and matches the sky clear colour.
    const float f = smoothstep(pc.fog.a * 0.55, pc.fog.a, inDistance);
    outColor = vec4(mix(albedo * light, pc.fog.rgb, f), 1.0);
}
