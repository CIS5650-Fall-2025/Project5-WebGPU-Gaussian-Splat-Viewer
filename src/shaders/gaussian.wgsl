struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>,
    fov: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
    @location(0) color: vec4<f32>,
    @location(1) conic: vec3<f32>,
    @location(2) opacity: f32,
    @location(3) ndc_center: vec2<f32>
};

struct Splat {
    //TODO: store information for 2D splat rendering
    ndc_pos_N_radius: array<u32,2>,
    color: array<u32,2>,
    conic_N_opacity: array<u32,2>
};

@group(0) @binding(0) var<storage, read> splats: array<Splat>;
@group(0) @binding(1) var<uniform> gs_multiplier: f32;
@group(0) @binding(2) var<uniform> camera: CameraUniforms;
@group(0) @binding(3) var<storage, read> sort_indices: array<u32>;
@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let splat_index = sort_indices[instance_index];
    let ndc_pos_xy = unpack2x16float(splats[splat_index].ndc_pos_N_radius[0]);
    let ndc_pos_zNradius = unpack2x16float(splats[splat_index].ndc_pos_N_radius[1]);
    let radius = ndc_pos_zNradius.y;
    let ndc_center = vec2<f32>(ndc_pos_xy.x, ndc_pos_xy.y);

    let base_scale = radius;
    let quad_width = base_scale / camera.viewport.x;
    let quad_height = base_scale / camera.viewport.y;
    let quad_offsets = array<vec2<f32>, 6>(
        vec2<f32>(-quad_width, -quad_height),  // First triangle
        vec2<f32>( quad_width, -quad_height),
        vec2<f32>( quad_width,  quad_height),
        vec2<f32>(-quad_width, -quad_height),  // Second triangle
        vec2<f32>( quad_width,  quad_height),
        vec2<f32>(-quad_width,  quad_height),
    );

    let vertex_pos = ndc_center + quad_offsets[vertex_index];

    let color_xy = unpack2x16float(splats[splat_index].color[0]);
    let color_z0 = unpack2x16float(splats[splat_index].color[1]);
    let color = vec3<f32>(color_xy.x, color_xy.y, color_z0.x);

    let conic_xy = unpack2x16float(splats[splat_index].conic_N_opacity[0]);
    let conic_z_N_opacity = unpack2x16float(splats[splat_index].conic_N_opacity[1]);
    let conic = vec3<f32>(conic_xy.x, conic_xy.y, conic_z_N_opacity.x);
    let opacity = conic_z_N_opacity.y;

    var out: VertexOutput;
    out.position = vec4<f32>(vertex_pos, 0., 1.);
    out.color = vec4<f32>(color, 1.);
    out.conic = conic;
    out.opacity = opacity;
    out.ndc_center = ndc_center;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ndc_center = in.ndc_center;
    let color = in.color;
    let conic = in.conic;
    let opacity = in.opacity;
    var pos_xy = in.position.xy;

    // return color;

    // compute offset from center
    pos_xy = pos_xy / camera.viewport * 2.0 - 1.0;
    pos_xy.y *= -1.0;
    var offset = (pos_xy - ndc_center) * 0.5 * camera.viewport;
    offset.x *= -1.0;


    let power = -0.5 * (conic.x * offset.x * offset.x + conic.z * offset.y * offset.y) 
                      - conic.y * offset.x * offset.y;

    if (power > 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    let alpha = min(0.99, opacity * exp(power));
    
    return color * alpha;
}