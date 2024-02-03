#version 450

layout(location = 0) in float densesy;
layout(location = 0) out vec4 outColor;

void main() {
    

    float a = (0.5-length(gl_PointCoord-0.5))*2;
    outColor = vec4(1, densesy, densesy, a);
    
}