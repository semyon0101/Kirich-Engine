#version 450

struct ParticleType{
    float rmin;
    float e;
    float m;
};

layout (set = 0, binding = 0) uniform ParameterUBO {
    ParticleType particleTypes[3];
    mat4 model;
    mat4 view;
    mat4 proj;
    int width;
    int height;
    int particleCount;
    int particleDivision;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inLPosition;
layout(location = 2) in int inType;
layout(location = 0) out float densesy;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1);
    //gl_PointSize = ubo.particleTypes[uint(inType)].rmin*1.5; 
    //gl_Position = vec4((inPosition - vec2(ubo.width, ubo.height) / 2) / vec2(ubo.width, ubo.height) * 2, 0, 1);
    gl_PointSize=1000/gl_Position.z*ubo.particleTypes[uint(inType)].rmin*1.5;

    densesy = length(inPosition-inLPosition)*3;
    //densesy=1;
}