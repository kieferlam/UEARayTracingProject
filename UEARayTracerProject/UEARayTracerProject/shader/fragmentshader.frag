#version 430
precision mediump float;

in vec2 texcoord;

out vec4 colour;

uniform sampler2D textureSampler;

void main(){
    colour = texture(textureSampler, texcoord);
}