#version 450

layout(input_attachment_index = 0, binding = 1, set = 1) uniform subpassInput inputDepthP1; 

layout(location = 0) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
    
    vec2 coord = gl_PointCoord - vec2(0.5);
    if (subpassLoad(inputDepthP1).r>gl_FragCoord.z){
        outColor = vec4(fragColor, 0.5 - length(coord));
    }else{
        outColor = vec4(0);
    }
//    if (coord.x>0){
//        outColor = vec4(fragColor, 1);
//    }else {
//        outColor = vec4(fragColor, (0.5+coord.x)*2);
//    }
}