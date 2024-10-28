struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
};

struct Splat {
    //TODO: information defined in preprocess compute shader
    screen_pos: vec3<f32>,
};

@group(0) @binding(0) var<storage, read> splats: array<Splat>;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let quad_offsets = array<vec2<f32>, 6>(
        vec2<f32>(-0.005, -0.005),  // First triangle
        vec2<f32>( 0.005, -0.005),
        vec2<f32>( 0.005,  0.005),
        vec2<f32>(-0.005, -0.005),  // Second triangle
        vec2<f32>( 0.005,  0.005),
        vec2<f32>(-0.005,  0.005),
    );
    let quad_center = splats[instance_index].screen_pos;

    let vertex_pos = quad_center.xy + quad_offsets[vertex_index];

    var out: VertexOutput;
    out.position = vec4<f32>(vertex_pos, 0., 1.);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0., 1., 0., 1.);
}