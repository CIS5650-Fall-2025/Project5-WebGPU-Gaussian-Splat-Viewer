struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
};

struct Splat {
    pos: vec2<f32>
};

@group(0) @binding(0) var<storage, read> splats: array<Splat>;
@group(0) @binding(1) var<storage, read> sort_indices : array<u32>;

@vertex
fn vs_main(
    @builtin(instance_index) instanceIdx: u32,
    @builtin(vertex_index) vertIdx: u32
) -> VertexOutput {
    var out: VertexOutput;

    let splatIdx = sort_indices[instanceIdx];
    let splat = splats[instanceIdx];

    // Draw splat as a quad (each vertex is the splat position + width/height offset)
    // Works because splat position was projected to NDC space in the preprocess step.    
    let quadVerts = array<vec2f, 6>(
        vec2f(splat.pos.x - 0.01, splat.pos.y + 0.01),
        vec2f(splat.pos.x - 0.01, splat.pos.y - 0.01),
        vec2f(splat.pos.x + 0.01, splat.pos.y - 0.01),
        vec2f(splat.pos.x + 0.01, splat.pos.y - 0.01),
        vec2f(splat.pos.x + 0.01, splat.pos.y + 0.01),
        vec2f(splat.pos.x - 0.01, splat.pos.y + 0.01)
    let viewSpacePos = (camera.view * vec4<f32>(pos, 1.0)).xyz;
    let viewSpacePos = (camera.view * vec4<f32>(pos, 1.0)).xyz;
    );

    out.position = vec4f(quadVerts[vertIdx], 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1.);
}