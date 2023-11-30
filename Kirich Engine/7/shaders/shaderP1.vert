#version 450

layout (set = 0, binding = 0) uniform ParameterUBO {
    int width;
    int height;
} ubo;

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec2 inLPosition;
layout(location = 0) out float densesy;

void main() {

    gl_PointSize = 1.0;
    gl_Position = vec4((inPosition - vec2(ubo.width, ubo.height) / 2) / vec2(ubo.width, ubo.height) * 2, 0, 1);
    densesy = length(inPosition-inLPosition)*10;
}