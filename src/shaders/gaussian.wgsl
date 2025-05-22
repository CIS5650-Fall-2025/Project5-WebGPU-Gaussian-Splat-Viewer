struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
};

struct Splat {
    //TODO: information defined in preprocess compute shader
    xyPacked: u32
};

@group(0) @binding(0)
var<storage, read_write> splats: array<Splat>;

// half the edge length of our splat quad in NDC
let HALF_SIZE: f32 = 0.005

// precomputed corner offsets for a 6-vertex quad
let CORNER_OFFSETS: array<vec2f, 6> = array<vec2f, 6>(
    vec2f(-HALF_SIZE,  HALF_SIZE),
    vec2f(-HALF_SIZE, -HALF_SIZE),
    vec2f( HALF_SIZE, -HALF_SIZE),
    vec2f( HALF_SIZE, -HALF_SIZE),
    vec2f( HALF_SIZE,  HALF_SIZE),
    vec2f(-HALF_SIZE,  HALF_SIZE)
);

@vertex
fn vs_main(
    @builtin(instance_index) instanceIndex: u32,
    @builtin(vertex_index) vertexIndex: u32
) -> VertexOutput {
    var out: VertexOutput;

    // unpack center XY from 2×16-bit floats
    let packed = splats[instanceIndex].xyPacked;
    let center: vec2f = unpack2x16float(packed);

    // apply corner offset
    let offset: vec2f = CORNER_OFFSETS[vertexIndex];
    let ndcPos: vec2f = center + offset;
    
    out.position = vec4f(ndcPos.x, ndcPos.y, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1.);
}