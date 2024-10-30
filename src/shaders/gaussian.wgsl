struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) conic: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) opacity: f32,
    @location(3) center: vec2<f32>,
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
    packedPos: u32,
    packedSize: u32,
    packedColor: array<u32,2>,
    packedConicAndOpacity: array<u32,2>
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<storage, read> splats: array<Splat>;
@group(0) @binding(2)
var<storage, read> sort_indices : array<u32>;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    let sorted_index = sort_indices[instance_index];
    let splat = splats[sorted_index];

    let pos = vec2<f32>(unpack2x16float(splat.packedPos));
    let radius = vec2<f32>(unpack2x16float(splat.packedSize));
    let opacity = unpack2x16float(splat.packedConicAndOpacity[1]).y;
    let color = vec4<f32>(
        unpack2x16float(splat.packedColor[0]),
        unpack2x16float(splat.packedColor[1])
    );
    let conic = vec3<f32>(
        unpack2x16float(splat.packedConicAndOpacity[0]),
        unpack2x16float(splat.packedConicAndOpacity[1]).x
    );

    let quad = array<vec2<f32>, 6>(
        vec2<f32>(pos.x - radius.x, pos.y + radius.y),
        vec2<f32>(pos.x - radius.x, pos.y - radius.y),
        vec2<f32>(pos.x + radius.x, pos.y - radius.y),
        vec2<f32>(pos.x + radius.x, pos.y - radius.y),
        vec2<f32>(pos.x + radius.x, pos.y + radius.y),
        vec2<f32>(pos.x - radius.x, pos.y + radius.y)
    );

    var out: VertexOutput;
    out.position = vec4<f32>(quad[vertex_index], 0.0, 1.0); // NDC
    out.conic = conic;
    out.color = color;
    out.opacity = opacity;
    out.center = pos; // Pixel space

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var position = in.position.xy / camera.viewport * 2.0 - 1.0; // NDC
    position.y = -position.y;

    var d = (position - in.center) * camera.viewport * 0.5; // Pixel space
    d.x = -d.x;

    let power = -0.5 * (in.conic.x * d.x * d.x + in.conic.z * d.y * d.y)
                - in.conic.y * d.x * d.y;
    if (power > 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let alpha = min(0.99f, in.opacity * exp(power));
    if (alpha < 1.0 / 255.0) {
        discard;
    }

    return in.color * alpha;
}