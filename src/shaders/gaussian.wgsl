
const quad_verts = array<vec2<f32>, 6>(
    vec2<f32>(-0.01, -0.01),  // Bottom-left
    vec2<f32>(0.01, -0.01),  // Bottom-right
    vec2<f32>(-0.01,  0.01),  // Top-left
    vec2<f32>(0.01, -0.01),  // Bottom-right
    vec2<f32>( 0.01,  0.01),  // Top-right
    vec2<f32>(-0.01,  0.01),  // Top-left
    );

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) v_color: vec3<f32>,
};



struct Splat {
    //TODO: information defined in preprocess compute shader
    position: vec2<f32>, 
    size: f32,          
    color: vec3<f32>, 
};



@group(0) @binding(0)var<storage, read> splats: array<Splat>;


@vertex
fn vs_main(
    @builtin(instance_index) instanceIndex: u32,
    @builtin(vertex_index) vert_idx : u32,) -> VertexOutput {
    var out: VertexOutput;

    var splat = splats[instanceIndex];
    //let size = splat.size;

    let offset = quad_verts[vert_idx];
    splat.position = vec2<f32>(splat.position.x + offset.x, splat.position.y + offset.y);

    out.position = vec4<f32>(splat.position, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}