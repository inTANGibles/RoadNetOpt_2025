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

uniform sampler2D inputTexture;
uniform vec2 direction;

uniform float blurRadius;

const int M = 16;
const int N = 2 * M + 1;

float getGaussianWeight(float x, float sigma)
{
    return exp(-x * x / (2.0 * sigma * sigma)) / (sqrt(2.0 * 3.141592653589793) * sigma);
}

void main()
{
    vec4 sum = vec4(0.0);

    for (int i = 0; i < N; ++i)
    {
        float weight = getGaussianWeight(float(i - M), blurRadius/3);
        vec2 tc = texcoord + direction * float(i - M);
        tc = clamp(tc, 0.01, 0.99);
        sum += weight * texture(inputTexture, tc);
    }

    out_color = clamp(sum, 0.0, 1.0);
}
#endif
