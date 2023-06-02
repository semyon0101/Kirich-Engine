#version 450 core

layout(input_attachment_index = 0, binding = 0, set = 0) uniform subpassInput inputColorP1; 
layout(input_attachment_index = 1, binding = 1, set = 0) uniform subpassInput inputDepthP1; 
layout(input_attachment_index = 2, binding = 2, set = 0) uniform subpassInput inputColorP2;
layout(input_attachment_index = 3, binding = 3, set = 0) uniform subpassInput inputDepthP2;

layout(location = 0) out vec4 outColor;

void main()
{
	//outColor = vec4(subpassLoad(inputColorP1).rgb*(1-subpassLoad(inputColorP2).a)+subpassLoad(inputColorP2).rgb,1);
	outColor = vec4(subpassLoad(inputColorP2).a);
}