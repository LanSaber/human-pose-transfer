#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform vec3 fragColor;

// texture samplers
uniform sampler2D texture1;

void main()
{
	// linearly interpolate between both textures (80% container, 20% awesomeface)
	//FragColor = texture(texture1, TexCoord);

    // vec4 texColor = texture(texture1, TexCoord);
      // if(texColor.a < 0.1)
       // discard;

	// FragColor = vec4(fragColor,1.0);
	FragColor = texture(texture1, TexCoord);
}