#version 450

layout (set = 0, binding = 0) uniform ParameterUBO {
    int division;
    int width;
    int height;
} ubo;

layout(location = 0) in vec2 inPosition;

void main() {

    gl_PointSize = 2.0;
    gl_Position = vec4((inPosition - vec2(ubo.width, ubo.height) / 2) / vec2(ubo.width, ubo.height) * 2, 0, 1);
}