#version 450 core

layout(input_attachment_index = 0, binding = 0) uniform subpassInput inputColor; 
layout(input_attachment_index = 1, binding = 1) uniform subpassInput inputDepth; 

layout(location = 0) out vec4 outColor;

void main()
{
//    int xHalf = 1280 / 2;
//    if (gl_FragCoord.x > xHalf)
//    {
//        float depth = subpassLoad(inputDepth).r;
//        float gray = 1.0 - depth;
//        color = vec4(gray, gray, gray, 1.0);
//    }
//    else
//        color = subpassLoad(inputColor).rgba;
	outColor = subpassLoad(inputColor).rgba;
}