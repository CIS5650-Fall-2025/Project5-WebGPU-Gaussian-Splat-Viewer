struct CameraUniforms {
    view: mat4x4f,
    view_inv: mat4x4f,
    proj: mat4x4f,
    proj_inv: mat4x4f,
    viewport: vec2f,
    focal: vec2f
};

struct Gaussian {
    pos_opacity: array<u32, 2>,
    rot: array<u32, 2>,
    scale: array<u32, 2>
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0)
var<storage, read> gaussians: array<Gaussian>;

struct VertexOutput {
    @builtin(position) position: vec4f,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32
) -> VertexOutput {
    var out: VertexOutput;

    let gaussian = gaussians[vertex_index];
    let a = unpack2x16float(gaussian.pos_opacity[0]);
    let b = unpack2x16float(gaussian.pos_opacity[1]);
    let pos = vec4f(a.x, a.y, b.x, 1.);

    out.position = camera.proj * camera.view * pos;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    return vec4f(1., 1., 0., 1.);
}