struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) size: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) conic_opacity: vec4<f32>,
    @location(3) center: vec2<f32>
};

struct Splat {
    xy: u32,
    widthHeight: u32,
    packed_color: array<u32, 2>,
    co: u32,
    cp: u32
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

    // unpack center XY from 2×16-bit floats
    let splatIdx = sort_indices[instanceIndex];
    let splat = splats[splatIdx];
    let xy = unpack2x16float(splat.xy);
    let size: vec2f = unpack2x16float(splat.widthHeight);

    // apply corner offset
    let corners = array<vec2f, 6>(
        vec2f(xy.x - size.x, xy.y + size.y),
        vec2f(xy.x - size.x, xy.y - size.y),
        vec2f(xy.x + size.x, xy.y - size.y),
        vec2f(xy.x + size.x, xy.y - size.y),
        vec2f(xy.x + size.x, xy.y + size.y),
        vec2f(xy.x - size.x, xy.y + size.y)
    );
    let corner = corners[vertexIndex];
    out.position = vec4f(corner.x, corner.y, 0.0, 1.0);
    out.size = size;

    let conicA = unpack2x16float(splat.co);
    let conicB = unpack2x16float(splat.cp);

    out.conic_opacity = vec4f(
        conicA.x,
        conicA.y,
        conicB.x,
        conicB.y
    );
    out.center = xy;

    let color = unpack2x16float(splat.packed_color[0]);
    out.color = vec4<f32>(unpack2x16float(splat.packed_color[0]),
                      unpack2x16float(splat.packed_color[1]));

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let windowPos = in.position.xy;
    let ndc = ((windowPos / camera.viewport) * 2.0 - vec2f(1.0)) * vec2f(1.0, -1.0);

    let d = ndc - in.center;
    var d_flipped = vec2f(-d.x, d.y);
    d_flipped *= camera.viewport * 0.5f;

    let conic = in.conic_opacity;
    let exponent = -0.5 * (
        conic.x * d_flipped.x * d_flipped.x +
        conic.z * d_flipped.y * d_flipped.y +
        2.0 * conic.y * d_flipped.x * d_flipped.y
    );

    if (exponent > 0.0f) {
        return vec4f(0.0f, 0.0f, 0.0f, 0.0f);
    }

    let alpha = min(0.99f, conic.w * exp(exponent));
    return in.color * alpha;
}