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

uniform sampler2D bound_texture;
uniform sampler2D region_texture;
uniform sampler2D building_texture;
uniform sampler2D raw_roads_texture;
uniform sampler2D new_roads_texture;
uniform sampler2D node_texture;

vec4 stack_color(vec4 color1, vec4 color2) {
    //stack two layers using alpha
    //color1 is at bottom and color2 is on top
    //return (1.0 - color2.a) * color1 + color2.a * color2;

    float a = 1 - (1 - color1.a) * ( 1 - color2.a);
    vec3 rgb = (1.0 - color2.a) * color1.rgb + color2.a * color2.rgb;
    vec4 rgba = vec4(rgb, a);
    return rgba;
}

void main()
{
    //handle bound
    vec4 bound_color = texture(bound_texture, texcoord);
    // inside bound : (1, 1, 1, 1)
    // outside bound : (0, 0, 0, 0)
    if (bound_color.a == 0.0){
        out_color = vec4(0.0, 1.0, 1.0, 1.0); // if not in bound region
        return; // simply return (0, 1, 1)
    }

    //handle nodes
//    vec4 nodes_color = texture(node_texture, texcoord);
//    if (nodes_color.a == 1.0){
//        out_color = vec4(0.0, 1.0, 1.0, 1.0);
//        return;// simply return (0, 1, 1)
//    }
    //handle region and building
    vec4 region_color = texture(region_texture, texcoord);
    vec4 building_color = texture(building_texture, texcoord);
    vec4 region_and_building_color = stack_color(region_color, building_color);
    // handle nodes color
    vec4 nodes_color = texture(node_texture, texcoord);
    if(nodes_color.a == 1.0){
        nodes_color = vec4(0.0, 1.0, 1.0, 1.0);
    }else{
        nodes_color = vec4(0.0, 0.0, 0.0, 0.0);
    }

    // handle roads
    vec4 raw_roads_color = texture(raw_roads_texture, texcoord);
    vec4 new_roads_color = texture(new_roads_texture, texcoord);
    vec4 roads_color = stack_color(raw_roads_color, new_roads_color);

    //handle final color
    //vec4 final_color = region_and_building_color + roads_color;
    vec4 color = stack_color(region_and_building_color, nodes_color);
    vec4 final_color = stack_color(color, roads_color);
    out_color = clamp(final_color, 0.0, 1.0);
}
#endif
