
struct VertexInput {
    
    @builtin(instance_index) instanceIndex: u32,
};

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
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let splat = splats[in.instanceIndex];
    let size = splat.size;
 
    return out;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>( 1.0);
}