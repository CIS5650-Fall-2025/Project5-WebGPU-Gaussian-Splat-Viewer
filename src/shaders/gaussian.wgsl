struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) wh: vec2f,
    @location(1) color: vec4f,
    @location(2) conic_opa: vec4f,
    @location(3) center: vec2f
};

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct Splat {
    xy: u32,
    wh: u32,
    rg: u32,
    ba: u32,
    co: u32,
    cp: u32
};

@group(0) @binding(0)
var<storage, read> splats: array<Splat>;
@group(0) @binding(1)
var<storage, read> sort_indices : array<u32>;
@group(0) @binding(2)
var<uniform> camera: CameraUniforms;

@vertex
fn vs_main(
    @builtin(instance_index) instanceIndex: u32,
    @builtin(vertex_index) vertexIndex: u32
) -> VertexOutput {
    var out: VertexOutput;

    let index = sort_indices[instanceIndex];
    let splat = splats[index];
    let xy = unpack2x16float(splat.xy);
    let wh = unpack2x16float(splat.wh);

    let corners = array<vec2f, 6>(
        vec2f(xy.x - wh.x, xy.y + wh.y),
        vec2f(xy.x - wh.x, xy.y - wh.y),
        vec2f(xy.x + wh.x, xy.y - wh.y),
        vec2f(xy.x + wh.x, xy.y - wh.y),
        vec2f(xy.x + wh.x, xy.y + wh.y),
        vec2f(xy.x - wh.x, xy.y + wh.y),
    );
    let pos = vec4(corners[vertexIndex].x, corners[vertexIndex].y, 0, 1);

    let rg = unpack2x16float(splat.rg);
    let ba = unpack2x16float(splat.ba);
    let color = vec4f(rg.x, rg.y, ba.x, ba.y);

    let co = unpack2x16float(splat.co);
    let cp = unpack2x16float(splat.cp);
    let conic_opa = vec4f(co.x, co.y, cp.x, cp.y);

    out.position = pos;
    out.wh = wh;
    out.color = color;
    out.conic_opa = conic_opa;
    out.center = xy;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var xy = (in.position.xy / camera.viewport) * 2.0f - 1.0f;
    xy.y *= -1.0f;

    var d = xy - in.center;
    d.x *= -1.0f;
    d *= camera.viewport * 0.5f;

    let power = -0.5f * (in.conic_opa.x * d.x * d.x + in.conic_opa.z * d.y * d.y) - in.conic_opa.y * d.x * d.y;

    if (power > 0.0f) {
        return vec4f(0.0f, 0.0f, 0.0f, 0.0f);
    }
    
    let alpha = min(0.99f, in.conic_opa.w * exp(power));
    
    return in.color * alpha;
    //return vec4<f32>(in.wh.x, in.wh.y, 0.0, 1.0);
    //return in.color;
}