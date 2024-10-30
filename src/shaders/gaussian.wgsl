struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) opacity: f32,
    @location(1) color: vec4<f32>,
    @location(2) midpoint: vec2<f32>,
    @location(3) inv_covar_2d: vec3<f32>,
};

struct Splat {
    //TODO: store information for 2D splat rendering
    pos_opacity: array<u32,2>,
    color_rg_ba: array<u32,2>,
    inv_covar_2d: array<u32,2>,
    radius: u32,
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
var<uniform> camera: CameraUniforms;

@group(0) @binding(1)
var<storage,read> splats: array<Splat>;

@group(0) @binding(2)
var<storage, read> sort_indices: array<u32>;

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {

    let dummy1 = camera.proj;
    var out: VertexOutput;

    let ind = sort_indices[instanceIndex];
    let splat = splats[ind];

    let pos_xy = unpack2x16float(splat.pos_opacity[0]);
    let pos_zo = unpack2x16float(splat.pos_opacity[1]);
    let color = vec4<f32>(
        unpack2x16float(splat.color_rg_ba[0]),
        unpack2x16float(splat.color_rg_ba[1])
    );
    let inv_covar_2d = vec3<f32>(
        unpack2x16float(splat.inv_covar_2d[0]).x,
        unpack2x16float(splat.inv_covar_2d[0]).y,
        unpack2x16float(splat.inv_covar_2d[1]).y
    );
    let position_ndc = vec3<f32>(pos_xy.x, pos_xy.y, pos_zo.x);
    let opacity = pos_zo.y;
    //let radius = splat.radius * (1 / 1000.0f);
    //let radius = 0.01;
    let rad_2d = unpack2x16float(splat.radius);

    // Define quad offsets for the four vertices. Use radius to include >97% of caussian CDF 
    var quadOffsets = array<vec2<f32>, 4>(
        vec2<f32>(-rad_2d.x, -rad_2d.y),
        vec2<f32>(rad_2d.x, -rad_2d.y),
        vec2<f32>(-rad_2d.x, rad_2d.y),
        vec2<f32>(rad_2d.x, rad_2d.y),
    );

    let quadOffset = quadOffsets[vertexIndex];
    //let position_pixels = vec2<f32>((position_ndc.x * 0.5 + 0.5) * camera.viewport.x, (position_ndc.y * 0.5 + 0.5) * camera.viewport.y);
    out.position = vec4<f32>(
        position_ndc.xy + quadOffset,
        0,
        1.0
    );

    // Pass opacity to the fragment shader
    out.opacity = opacity;
    out.color = color;
    out.inv_covar_2d = vec3f(inv_covar_2d.x, inv_covar_2d.y, inv_covar_2d.z);
    out.midpoint = position_ndc.xy;
    out.midpoint.y = -out.midpoint.y; // flip y axis
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dummy = camera.proj;
    let delta = ((in.position.xy / camera.viewport.xy) * 2.0 - 1.0) - in.midpoint;
    let delta_pix = vec2<f32>(0.5 * delta.x * camera.viewport.x, 0.5 * delta.y * camera.viewport.y);
    let expon = -(0.5 * delta_pix.x * delta_pix.x * in.inv_covar_2d.x + delta_pix.x * delta_pix.y * in.inv_covar_2d.y + 0.5 * delta_pix.y * delta_pix.y * in.inv_covar_2d.z);
    //let expon = 0.0f;
    if expon > 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let final_a = min(0.99, in.opacity * exp(expon));

    return vec4<f32>(in.color.x, in.color.y, in.color.z, final_a); //* final_a;//in.opacity);
}