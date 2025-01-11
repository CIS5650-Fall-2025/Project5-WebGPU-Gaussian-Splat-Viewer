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
}

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0)
var<storage, read> gaussians: array<Gaussian>;

struct VertexOutput {
    @builtin(position) position: vec4f,
    //TODO: information passed from vertex shader to fragment shader
};

// struct Splat {
//     //TODO: information defined in preprocess compute shader
// };

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
    @builtin(vertex_index) in_vertex_index: u32
) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass 
    // var out: VertexOutput;

    const vertices = array<vec2f, 4>(
        vec2f(-0.01, -0.01),
        vec2f( 0.01, -0.01),
        vec2f(-0.01,  0.01),
        vec2f( 0.01,  0.01)
    );

    let vertex = gaussians[in_vertex_index];

    let pos_opacity0 = unpack2x16float(vertex.pos_opacity[0]);
    let pos_opacity1 = unpack2x16float(vertex.pos_opacity[1]);
    let rotation0    = unpack2x16float(vertex.rot[0]);
    let rotation1    = unpack2x16float(vertex.rot[1]);
    let scale0       = unpack2x16float(vertex.scale[0]);
    let scale1       = unpack2x16float(vertex.scale[1]);

    let pos = vec4f(pos_opacity0.x, pos_opacity0.y, pos_opacity1.x, 1.);
    let opacity = pos_opacity1.y;
    let rotation = vec4f(rotation0.xy, rotation1.xy);
    var scale = vec4f(scale0.xy, scale1.xy);

    // out.position = vec4f(1. ,1. , 0., 1.);
    return vec4f(vertices[in_vertex_index], 0.0, 1.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    return vec4f(1., 1., 1., 1, 1.);
}