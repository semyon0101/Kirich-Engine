#version 450

layout (binding = 0) uniform ParameterUBO {
    int fps;
    int width;
    int height;
    float randNumber;
} ubo;

layout(input_attachment_index = 0, binding = 0, set = 1) uniform subpassInput inputColorP1; 
layout (rgba8, binding = 0, set = 2) uniform readonly image2D inputImage;
layout (rgba8, binding = 1, set = 2) uniform image2D outputImage;
layout(location = 0) out vec4 outColor;

void main() {
    ivec2 uv = ivec2(gl_FragCoord);
    float strength=0;
    float dps = 1000;
    float cps = 200;
    float pps = 1000;

    float r = imageLoad(inputImage, uv).r + dps/ubo.fps;
    if (r > 1){
        r -= 1;
        strength += (imageLoad(inputImage, uv).a * 256)*0.1;
        int radiys = 1;
        for(int x= -radiys; x<=radiys; x+=1){
            for(int y= -radiys; y<=radiys; y+=1){
                if (x==0 && y==0) continue;
                strength += (imageLoad(inputImage, uv + ivec2(x, y)).a * 256)*0.9/8;
                 
            }
        }
    }else{
        strength = imageLoad(inputImage, uv).a * 256;
    }

    float b = imageLoad(inputImage, uv).b + cps/ubo.fps;
    if (b > 1){
        b -= 1;
        strength -= 1;
        if (strength < 0) strength = 0;
    }

    float g = imageLoad(inputImage, uv).g + pps/ubo.fps;
    if (g > 1){
        g -= 1;
        strength += subpassLoad(inputColorP1).a * 256;
    
    }
    
    

    
    imageStore(outputImage, uv, vec4(r, g, b, strength/256));

    outColor = vec4(1-strength/256, 1-strength/256, 1-strength/256, 1);
    
}