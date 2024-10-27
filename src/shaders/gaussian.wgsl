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

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) conic_matrix_0: vec3<f32>,
    @location(1) conic_matrix_1: vec3<f32>,
    @location(2) conic_matrix_2: vec3<f32>,
    @location(3) color: vec4<f32>
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<storage, read> sort_indices : array<u32>;
@group(0) @binding(2)
var<storage, read> splats: array<Splat>;

@vertex
fn vs_main(
) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass 

    let splat = splats[sort_indices[instance_index]];

    let mean_xy = unpack2x16float(splat.mean_xy);
    let diameter = 2.0 * unpack2x16float(splat.radii);
    let rgb_rg = unpack2x16float(splat.rgb_rg);
    let rgb_b_opacity = unpack2x16float(splat.rgb_b_opacity);;
    let conic_xy = unpack2x16float(splat.conic_xy);
    let conic_z = unpack2x16float(splat.conic_z);

    out.position = vec4<f32>(vertex_pos_ndc, splat.pos_ndc.z, 1.0);

    out.mean_xy = mean_xy;
    out.radii = diameter/2;
    out.conic_xy = splat.conic_xy;
    out.rgb_rg = rgb_rg;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var pos_ndc = 2.0 * (input.position.xy / camera.viewport) - vec2(1.0, 1.0);
    pos_ndc.y = -pos_ndc.y;

    let conic_matrix = mat3x3<f32>(
        in.conic_matrix_0,
        in.conic_matrix_1,
        in.conic_matrix_2
    );

    let distance_from_center = dot(pos_ndc, conic_matrix * pos_ndc);

    if (distance_from_center > 1.0) {
        discard;
    }

    let decay = exp(-distance_from_center);

    return vec4<f32>(in.color.rgb, in.color.a * decay);
}