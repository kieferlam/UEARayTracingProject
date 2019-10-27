#version 430
precision float mediump;

in vec2 texcoord;

out vec4 colour;

uniform sampler2D textureSampler;

int main(){
    colour = texture(textureSampler, texcoord);
}