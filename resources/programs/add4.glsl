#version 330

#if defined VERTEX_SHADER

in vec3 in_position;
in vec2 in_texcoord_0;
out vec2 texcoord;
void main() {
    gl_Position = vec4(in_position, 1.0);
    texcoord = in_texcoord_0;
}


#elif defined FRAGMENT_SHADER
in vec2 texcoord;
layout (location = 0) out vec4 out_color;

uniform sampler2D inputTexture0;
uniform sampler2D inputTexture1;
uniform sampler2D inputTexture2;
uniform sampler2D inputTexture3;


void main()
{
	vec4 colors[4];
    colors[0] = texture(inputTexture0, texcoord);
    colors[1] = texture(inputTexture1, texcoord);
    colors[2] = texture(inputTexture2, texcoord);
    colors[3] = texture(inputTexture3, texcoord);

    vec4 final_color = vec4(0.0);
    for (int i = 0; i < 4; i++) {
        final_color += colors[i];
    }
    out_color = clamp(final_color, 0.0, 1.0);
}
#endif
