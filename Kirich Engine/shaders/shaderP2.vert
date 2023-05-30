#version 450

layout(binding = 0) uniform UniformBufferObject {
    float deltaTime;
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec4 inColor;

layout(location = 0) out vec3 fragColor;

void main() {

    gl_PointSize = 20.0;
    gl_Position = ubo.proj * ubo.view* ubo.model* vec4(inPosition, 1.0);
    fragColor = inColor.rgb;
}