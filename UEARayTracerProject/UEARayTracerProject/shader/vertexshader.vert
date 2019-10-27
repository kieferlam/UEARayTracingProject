#version 430

layout(location = 0) in vec4 position;
layout(location = 1) in vec2 textureCoord;

out vec2 texcoord;

void main(){
    texcoord = textureCoord;
    gl_Position = position;
}