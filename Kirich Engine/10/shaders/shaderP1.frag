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

layout(location = 0) in float densesy;
layout(location = 1) in float pointSize;
layout(location = 0) out vec4 outColor;

void main() {
    
    float a = (0.5-length(gl_PointCoord-0.5));
    if(a>0) {
        a=1;
        float r = sqrt(pointSize*pointSize-gl_PointCoord.x*gl_PointCoord.x-gl_PointCoord.y*gl_PointCoord.y);
        mat4x4 prj_mat=ubo.proj * ubo.view * ubo.model;
        float A = prj_mat[2][2];
        float B = prj_mat[3][2];
        gl_FragDepth=B / (B/gl_FragCoord.z-r/100000);
    }
    else gl_FragDepth=1;
    outColor = vec4(1, densesy, densesy, 1);
    
}
// depth=B / (A + z_ndc)
// z_ndc = B/depth-A
// depth = B / (B/depth-r)