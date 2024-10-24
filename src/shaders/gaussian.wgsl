struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) conic_matrix_0: vec3<f32>,
    @location(1) conic_matrix_1: vec3<f32>,
    @location(2) conic_matrix_2: vec3<f32>,
    @location(3) color: vec4<f32>
};

struct Splat {
    pos_ndc: vec3<f32>,
    size_ndc: vec2<f32>,
    conic_matrix: mat3x3<f32>,
    color: vec4<f32>
};

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

@group(0) @binding(0)
var<storage, read> splats: array<Splat>;

@vertex
fn vs_main(
) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass 
    var out: VertexOutput;
    out.position = vec4<f32>(1. ,1. , 0., 1.);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1.);
}