#version 450

layout(binding = 0) buffer UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec3 position;
layout(location = 1) out vec3 pos;

void main() {
    gl_Position = ubo.proj * ubo.view * vec4(position, 1.0);
    pos = gl_Position.xyz;
}
