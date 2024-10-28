struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
};

struct Splat {
    xy: vec2f,
    size: vec2f,
};

@group(0) @binding(0)
var<storage, read> sort_indices : array<u32>;
@group(0) @binding(1)
var<storage, read> splats: array<Splat>;

@vertex
fn vs_main(
    @builtin(instance_index) instance : u32,
    @builtin(vertex_index) vertex : u32
) -> VertexOutput {

    let splat = splats[instance];

    let positions = array(
        splat.xy + vec2(-splat.size.x,  splat.size.y),
        splat.xy + vec2(-splat.size.x, -splat.size.y),
        splat.xy + vec2( splat.size.x, -splat.size.y),

        splat.xy + vec2( splat.size.x, -splat.size.y),
        splat.xy + vec2( splat.size.x,  splat.size.y),
        splat.xy + vec2(-splat.size.x,  splat.size.y),
    );

    let sort_index = sort_indices[0];

    var out: VertexOutput;
    out.position = vec4(positions[vertex], 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1.);
}