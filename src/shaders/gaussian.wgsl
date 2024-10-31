struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct Splat {
    mean_xy: u32,
    radii: u32,
    conic_xy: u32,
    conic_z: u32,
    rgb_rg: u32,
    rgb_b_opacity: u32
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<storage, read> sort_indices : array<u32>;
@group(0) @binding(2)
var<storage, read> splats: array<Splat>;

const signs = array<vec2f, 6>(
    vec2f(-1.0, 1.0), vec2f(-1.0, -1.0),
    vec2f(1.0, -1.0), vec2f(1.0, -1.0),
    vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) mean: vec2f,
    @location(1) rgb_opacity: vec4f,
    @location(2) conic: vec3f
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let splat = splats[sort_indices[instance_index]];

    let mean_xy = unpack2x16float(splat.mean_xy);
    let diameter = 2.0 * unpack2x16float(splat.radii);
    let rgb_rg = unpack2x16float(splat.rgb_rg);
    let rgb_b_opacity = unpack2x16float(splat.rgb_b_opacity);;
    let conic_xy = unpack2x16float(splat.conic_xy);
    let conic_z = unpack2x16float(splat.conic_z);

    let sign = signs[vertex_index];

    return VertexOutput(
        vec4(mean_xy.x + sign.x*diameter.x, mean_xy.y + sign.y*diameter.y, 0.0, 1.0),
        vec2f(mean_xy.x, mean_xy.y),
        vec4f(rgb_rg.x, rgb_rg.y, rgb_b_opacity.x, 1.0/(1.0+exp(-rgb_b_opacity.y))),
        vec3f(conic_xy.x, conic_xy.y, conic_z.x),
    );
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    var pos_ndc = 2.0 * (input.position.xy / camera.viewport) - vec2(1.0, 1.0);
    pos_ndc.y = -pos_ndc.y;

    // Screen-space offset from the fragment position, with x-coordinate reversed.
    var offset_screen = pos_ndc - input.mean;
    offset_screen.x = -offset_screen.x;
    offset_screen *= camera.viewport * 0.5;

    var exponent = 
        input.conic.x * offset_screen.x * offset_screen.x
        + input.conic.z * offset_screen.y * offset_screen.y
        + input.conic.y * offset_screen.x * offset_screen.y;

    return vec4f(input.rgb_opacity.xyz, 1.0) * input.rgb_opacity.w * exp(-exponent/2.0);
}