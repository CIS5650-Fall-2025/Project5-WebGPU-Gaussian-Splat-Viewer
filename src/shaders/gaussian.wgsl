struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>,
    fov: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
    @location(0) color: vec4<f32>,
};

struct Splat {
    //TODO: store information for 2D splat rendering
    screen_pos: vec3<f32>,
    radius: f32,
};

@group(0) @binding(0) var<storage, read> splats: array<Splat>;
@group(0) @binding(1) var<uniform> gs_multiplier: f32;
@group(0) @binding(2) var<uniform> camera: CameraUniforms;
@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let base_scale = splats[instance_index].radius;
    let quad_width = base_scale / camera.viewport.x;
    let quad_height = base_scale / camera.viewport.y;
    let quad_offsets = array<vec2<f32>, 6>(
        vec2<f32>(-quad_width, -quad_height),  // First triangle
        vec2<f32>( quad_width, -quad_height),
        vec2<f32>( quad_width,  quad_height),
        vec2<f32>(-quad_width, -quad_height),  // Second triangle
        vec2<f32>( quad_width,  quad_height),
        vec2<f32>(-quad_width,  quad_height),
    );
    let quad_center = splats[instance_index].screen_pos;

    let vertex_pos = quad_center.xy + quad_offsets[vertex_index];

    var out: VertexOutput;
    out.position = vec4<f32>(vertex_pos, 0., 1.);
    out.color = vec4<f32>(quad_width, quad_height, 0., 1.);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}