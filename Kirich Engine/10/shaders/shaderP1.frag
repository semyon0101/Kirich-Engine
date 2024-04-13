#version 450
#define PARTICLE_PARAMETR_COUNT 3

struct ParticleType{
    float rmin;
    float e;
    float m;
};

layout (set = 0, binding = 0) uniform ParameterUBO {
    ParticleType particleTypes[PARTICLE_PARAMETR_COUNT];
    mat4 model;
    mat4 view;
    mat4 proj;
    int width;
    int height;
    int particleCount;
    int particleDivision;
} ubo;

layout(location = 0) in float densesy;
layout(location = 1) in float pointRadius;
layout(location = 2) in vec4 pointCenter;
layout(location = 3) in mat4 inverseMatrixs;
layout(location = 0) out vec4 outColor;

void main() {
    
    float a = (1-length(gl_PointCoord-0.5)*2);
    float al=0;
    if(a>0) {
        float r =sqrt(1-pow(1-a,2))*pointRadius;
        a=1;
        mat4x4 prj_mat=ubo.proj;
        float A = prj_mat[2][2];
        float B = prj_mat[3][2];
        gl_FragDepth = B / (B/(gl_FragCoord.z+A)-r)-A;
    }
    else gl_FragDepth=1;

    vec4 position = inverseMatrixs*vec4((gl_FragCoord.x/ubo.width-0.5)*2,(gl_FragCoord.y/ubo.height-0.5)*2, gl_FragDepth, 1);
    vec3 normal = normalize(pointCenter.xyz/pointCenter.w - position.xyz/position.w);

    vec3 sun = vec3(1/sqrt(3),1/sqrt(3),-1/sqrt(3));
    al=dot(sun,normal);
    al=(al+1)/4+0.2;
    //al=(al-0.2)*1.5;
    
    //if (al<0.5)al=0.5;
    outColor = vec4(al,densesy* al, densesy*al, 1);
}