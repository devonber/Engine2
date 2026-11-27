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

layout(location = 0) out vec4 outColor;

void main()
{
    // Fog hides the edge of the loaded world and matches the sky clear colour.
    const vec3 albedo = texture(blockTextures, vec3(inUV, float(inLayer))).rgb;
    const float f = smoothstep(pc.fog.a * 0.55, pc.fog.a, inDistance);
    outColor = vec4(mix(albedo * inShade, pc.fog.rgb, f), 1.0);
}
