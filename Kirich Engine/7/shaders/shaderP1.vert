#version 450

struct ParticleType{
    float rmin;
    float e;
    float m;
};

layout (set = 0, binding = 0) uniform ParameterUBO {
    ParticleType particleTypes[2];
    int width;
    int height;
} ubo;

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec2 inLPosition;
layout(location = 2) in int inType;
layout(location = 0) out float densesy;

void main() {
    
    gl_PointSize = ubo.particleTypes[uint(inType)].rmin*1.5;
    gl_Position = vec4((inPosition - vec2(ubo.width, ubo.height) / 2) / vec2(ubo.width, ubo.height) * 2, 0, 1);
    densesy = length(inPosition-inLPosition)*10;
}