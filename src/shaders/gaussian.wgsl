@group(0) @binding(0)
var<storage, read> splats: array<Splat>;
@group(0) @binding(1)
var<storage, read> sort_indices : array<u32>;
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
    //TODO: information defined in preprocess compute shader
    // Data is packed, need to unpack
    position_and_size: array<u32,2>,
    color_data: array<u32,2>,
    conic_and_opacity: array<u32,2>,
};

struct CameraUniforms {
    view: mat4x4f,
    view_inv: mat4x4f,
    proj: mat4x4f,
    proj_inv: mat4x4f,
    viewport: vec2f,
    focal: vec2f,
};

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32, @builtin(instance_index) in_instance_index: u32
) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass 
    var out: VertexOutput;

    let splat = splats[sort_indices[in_instance_index]];
    let xy = unpack2x16float(splat.position_and_size[0]);
    let wh = unpack2x16float(splat.position_and_size[1]);
    let rg = unpack2x16float(splat.color_data[0]);
    let ba = unpack2x16float(splat.color_data[1]);
    let conic = unpack2x16float(splat.conic_and_opacity[0]);
    let opacity = unpack2x16float(splat.conic_and_opacity[1]);
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