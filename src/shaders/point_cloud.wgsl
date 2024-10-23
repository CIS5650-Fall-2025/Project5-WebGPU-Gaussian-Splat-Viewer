struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

// On the CPU side, a Float16Array of 11 entries. Interpreted as follows:
struct Gaussian {
    // 4 16-bit floats for position (x,y,z) and opacity packed as 2 u32.
    pos_opacity: array<u32,2>,
    // 4 16-bit floats for rotation (x,y,z,w) packed as 2 u32.
    rot: array<u32,2>,
    // 3 16-bit floats for scale (x,y,z) packed as 2 u32.
    scale: array<u32,2>
}

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0)
var<storage,read> gaussians : array<Gaussian>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;

    let vertex = gaussians[in_vertex_index];
    let a = unpack2x16float(vertex.pos_opacity[0]);
    let b = unpack2x16float(vertex.pos_opacity[1]);
    var pos = vec4<f32>(a.x, a.y, b.x, 1.);

    // TODO: MVP calculations

    // Scale and rotation applies to the Gaussian splats at this point,
    // but not to the point itself.

    pos = camera.proj * camera.view * pos;
    out.position = pos;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1., 1., 0., 1.);
}