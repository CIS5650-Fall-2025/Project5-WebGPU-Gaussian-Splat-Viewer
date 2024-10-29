struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
    @location(0) color:vec3<f32>,
    @location(1) conic: vec2<f32>,
    @location(2) rec_minbound: vec2<f32>,
    @location(3) rec_maxbound: vec2<f32>,
    @location(4) radius:f32
};

struct Splat {
    //TODO: information defined in preprocess compute shader
    color: vec3<f32>,
    radius: f32,
    depths: f32,
    conic: vec2f,
    projpos: vec2f
};

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
}
@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0)
var<storage,read> gaussians: array<Gaussian>;
@group(1) @binding(1)
var<storage,read> splats: array<Splat>;

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass 
    var out: VertexOutput;
    out.position = vec4<f32>(1. ,1. , 0., 1.);
    let BLOCK_X = 16;
    let BLOCK_Y = 16; // tile sizes
    let blockSize = vec2<i32>(BLOCK_X, BLOCK_Y);
    // block count for x,y axes
    let grid = (camera.viewport.xy + vec2f(blockSize.xy) - 1) / vec2f(blockSize.xy);
    // out.position= vec4f(vec2f(splats[in_vertex_index].projpos),0.,1.);//uv
    let vertex = gaussians[in_vertex_index];
    let a = unpack2x16float(vertex.pos_opacity[0]);
    let b = unpack2x16float(vertex.pos_opacity[1]);
    let pos = vec4<f32>(a.x, a.y, b.x, 1.);

    let viewprojmat = camera.proj*camera.view;
    out.position = viewprojmat*pos;

    out.rec_minbound = vec2<f32>(
        (min(grid.x, max(0, ((out.position.x - splats[in_vertex_index].radius) / f32(BLOCK_X))))),
        (min(grid.y, max(0, ((out.position.y - splats[in_vertex_index].radius) / f32(BLOCK_Y)))))
    );  
    out.rec_maxbound = vec2<f32>(
        (min(grid.x, (max(0, ((out.position.x + splats[in_vertex_index].radius + f32(BLOCK_X) - 1) / f32(BLOCK_X)))))),
        (min(grid.y, (max(0, ((out.position.y + splats[in_vertex_index].radius + f32(BLOCK_Y) - 1) / f32(BLOCK_Y))))))
    );
    out.conic = splats[in_vertex_index].conic;
    out.color = splats[in_vertex_index].color;
    out.radius = splats[in_vertex_index].radius;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let closestPoint = clamp(in.position.xy,in.rec_minbound,in.rec_maxbound);
    let distance = length(closestPoint-in.position.xy);
    var color = vec4<f32>(0., 0., 0., 1.);
    if(distance <= in.radius)
    {
        color = f32(distance - in.radius)*color;
    }
    return color;
}