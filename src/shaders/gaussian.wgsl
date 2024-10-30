
@group(0) @binding(0)
var<storage,read> splats: array<Splat>;
@group(0) @binding(1)
var<storage,read> sort_indices:array<u32>;
@group(0) @binding(2)
var<uniform> camera: CameraUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
    @location(0) color: vec4f,
    @location(1) conic_opacity: vec4f,
    @location(2) center: vec2f,
};

struct Splat {
    //TODO: store information for 2D splat rendering
    color: array<u32,2>,
    radius_depths_pos: array<u32,2>,
    conic: array<u32,2>
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

// Quad vertices in NDC space (counter-clockwise)
const VERTICES = array<vec2<f32>, 6>(
    vec2f(-1.0, -1.0), 
    vec2f( 1.0, -1.0), 
    vec2f(-1.0,  1.0),
    vec2f(-1.0,  1.0), 
    vec2f( 1.0, -1.0), 
    vec2f( 1.0,  1.0)
);
@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,@builtin(instance_index) in_instance_index:u32
) -> VertexOutput {
    var out: VertexOutput;

    let splat = splats[sort_indices[in_instance_index]];
    let xy = unpack2x16float(splat.radius_depths_pos[1]);
    let wh = unpack2x16float(splat.radius_depths_pos[0]);
    let rg = unpack2x16float(splat.color[0]);
    let ba = unpack2x16float(splat.color[1]);
    let conic = unpack2x16float(splat.conic[0]);
    let opacity = unpack2x16float(splat.conic[1]);
    let pos = array<vec2f, 6>(
        vec2f(xy.x - wh.x * 2.0f, xy.y + wh.y * 2.0f),
        vec2f(xy.x - wh.x * 2.0f, xy.y - wh.y * 2.0f),
        vec2f(xy.x + wh.x * 2.0f, xy.y - wh.y * 2.0f),
        vec2f(xy.x + wh.x * 2.0f, xy.y - wh.y * 2.0f),
        vec2f(xy.x + wh.x * 2.0f, xy.y + wh.y * 2.0f),
        vec2f(xy.x - wh.x * 2.0f, xy.y + wh.y * 2.0f),
    );
    
    out.position = vec4(pos[in_vertex_index].x, pos[in_vertex_index].y, 0.0f, 1.0f);
    out.color = vec4(rg.x, rg.y, ba.x, ba.y);;
    out.conic_opacity = vec4f(conic.x, conic.y, opacity.x, opacity.y);
    out.center = vec2f(xy.x, xy.y);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var position = (in.position.xy / camera.viewport) * 2.0f - 1.0f;
    position.y *= -1.0f;
    
    var offset = (position.xy - in.center.xy) * camera.viewport * 0.5f;
    offset.x *= -1.0f;

    var power = -0.5f * (
        in.conic_opacity.x * pow(offset.x, 2.0f) + 
        in.conic_opacity.z * pow(offset.y, 2.0f)
    );
    power -= in.conic_opacity.y * offset.x * offset.y;

    if (power > 0.0f) {
        return vec4f(0.0f, 0.0f, 0.0f, 0.0f);
    }
    
    let alpha = min(0.99f, in.conic_opacity.w * exp(power));
    return in.color * alpha;
}
