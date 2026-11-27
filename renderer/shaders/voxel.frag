#version 460

layout(push_constant) uniform PushConstants {
    mat4 viewProj;
    vec4 cameraPos;
    vec4 fog; // rgb = colour (linear), a = distance of full fog
} pc;

layout(location = 0) flat in vec3 inColor;
layout(location = 1) in float inDistance;

layout(location = 0) out vec4 outColor;

void main()
{
    // Fog hides the edge of the loaded world and matches the sky clear colour.
    const float f = smoothstep(pc.fog.a * 0.55, pc.fog.a, inDistance);
    outColor = vec4(mix(inColor, pc.fog.rgb, f), 1.0);
}
