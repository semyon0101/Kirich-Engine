#version 450

layout(input_attachment_index = 0, binding = 1, set = 0) uniform subpassInput inputColorP2; 

layout(location = 0) out vec4 outColor;

void main()
{
	outColor = vec4(subpassLoad(inputColorP2).rgb, 1);
}