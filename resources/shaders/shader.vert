#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;
layout (location = 2) in vec3 aNormal;

out vec2 TexCoord;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float offset;

void main()
{
	// gl_Position = projection * view  *model* vec4(aPos[0],aPos[1], aPos[2], 1.0f);
	vec4 position = projection * view  *model* vec4(aPos[0],aPos[1], aPos[2], 1.0f);
	gl_Position = vec4(aPos[0],aPos[1], aPos[2], 1.0f);
	//gl_Position = temp.xyww;
	//gl_Position=vec4(aPos, 1.0f);
	TexCoord = vec2(aTexCoord.x, aTexCoord.y);
}