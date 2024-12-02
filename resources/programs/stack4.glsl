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

vec4 blend_images(vec4 colors[4], float alphas[4]) {
    vec4 final_color = colors[0];
    for (int i = 1; i < 4; i++) {
        final_color = (1.0 - alphas[i]) * final_color + alphas[i] * colors[i];
    }
    return final_color;
}

void main()
{
	vec4 colors[4];
    float alphas[4];


    colors[0] = texture(inputTexture0, texcoord);
    alphas[0] = colors[0].a;
    colors[1] = texture(inputTexture1, texcoord);
    alphas[1] = colors[1].a;
    colors[2] = texture(inputTexture2, texcoord);
    alphas[2] = colors[2].a;
    colors[3] = texture(inputTexture3, texcoord);
    alphas[3] = colors[3].a;

    vec4 final_color = blend_images(colors, alphas);

    out_color = final_color;
}
#endif
