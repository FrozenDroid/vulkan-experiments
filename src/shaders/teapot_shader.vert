#version 450

layout(set = 0, binding = 0) uniform sampler2D texSampler;
layout(set = 0, binding = 1) uniform sampler2D roughnessSampler;
layout(set = 0, binding = 2) uniform sampler2D emissiveSampler;
layout(set = 0, binding = 3) uniform sampler2D normalSampler;

layout(set = 1, binding = 0) buffer UniformBufferObject {
    vec3 cam_pos;
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normals;
layout(location = 2) in vec2 tex_coords;

layout(location = 0) out vec2 fragTexCoord;
layout(location = 1) out vec3 worldPos;
layout(location = 2) out vec3 normal;
layout(location = 3) out vec3 cam_pos;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(position, 1.0);
    fragTexCoord = tex_coords;
    worldPos = vec3(ubo.model * vec4(position, 1.0));
    normal = mat3(ubo.model) * normals;
    cam_pos = ubo.cam_pos;
}
