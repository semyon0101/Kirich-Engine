#version 450

layout(location = 0) out vec4 outColor;

void main() {
    //float a = (0.5-length(gl_PointCoord-0.5))*2;
    //a/=250;
    float a = (0.5-length(gl_PointCoord-0.5))*2;
    if (a>0.01){
        a = 1.0/256;
    }
    outColor = vec4(0, 0, 0, a);
    
}