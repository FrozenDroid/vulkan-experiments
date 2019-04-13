#version 450

layout(binding = 0) buffer UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec3 position;

void main() {
    gl_Position = ubo.proj * ubo.view * vec4(position, 1.0);
}
