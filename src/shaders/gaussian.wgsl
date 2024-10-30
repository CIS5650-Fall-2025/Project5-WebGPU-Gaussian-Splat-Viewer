


struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) v_color: vec3<f32>,
};



struct Splat {
    //TODO: information defined in preprocess compute shader
    position: vec2<f32>, 
    size: vec2<f32>,          
    color: vec3<f32>, 
};



@group(0) @binding(0)var<storage, read> splats: array<Splat>;
@group(0) @binding(1)var<storage, read> sorted_indices : array<u32>;


@vertex
fn vs_main(
    @builtin(instance_index) instanceIndex: u32,
    @builtin(vertex_index) vert_idx : u32,) -> VertexOutput {
    var out: VertexOutput;

    let index = sorted_indices[instanceIndex];
    var splat = splats[index];
    let size = splat.size;

    let quad_verts = array<vec2<f32>, 6>(
    vec2<f32>(-size.x, size.y),  
    vec2<f32>(-size.x, -size.y), 
    vec2<f32>(size.x,  -size.y),  
    vec2<f32>(size.x, -size.y),  
    vec2<f32>( size.x,  size.y),  
    vec2<f32>(-size.x,  size.y),  
    );



    let offset = quad_verts[vert_idx];
    let position = vec2<f32>(splat.position.x + offset.x, splat.position.y + offset.y);

    out.position = vec4<f32>(position, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}