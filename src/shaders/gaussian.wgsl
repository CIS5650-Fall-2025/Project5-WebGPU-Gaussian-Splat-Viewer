struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader

    @location(0) color: vec4f,
    @location(1) conic: vec3f,
    @location(2) opacity: f32,
    @location(3) center: vec2f
};

struct Splat {
    xy: vec2f,
    size: vec2f,

    packed_color: array<u32, 2>,
    packed_conic_opacity: array<u32, 2>
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
var<storage, read> sort_indices : array<u32>;
@group(0) @binding(1)
var<storage, read> splats: array<Splat>;
@group(0) @binding(2)
var<uniform> camera: CameraUniforms;

@vertex
fn vs_main(
    @builtin(instance_index) instance : u32,
    @builtin(vertex_index) vertex : u32
) -> VertexOutput {

    let splat_index = sort_indices[instance];
    let splat = splats[splat_index];

    let positions = array(
        splat.xy + vec2(-splat.size.x,  splat.size.y),
        splat.xy + vec2(-splat.size.x, -splat.size.y),
        splat.xy + vec2( splat.size.x, -splat.size.y),

        splat.xy + vec2( splat.size.x, -splat.size.y),
        splat.xy + vec2( splat.size.x,  splat.size.y),
        splat.xy + vec2(-splat.size.x,  splat.size.y),
    );

    let c_a = unpack2x16float(splat.packed_conic_opacity[0]);
    let c_b = unpack2x16float(splat.packed_conic_opacity[1]);

    var out: VertexOutput;
    out.position = vec4(positions[vertex], 0.0, 1.0);
    out.color = vec4(unpack2x16float(splat.packed_color[0]), unpack2x16float(splat.packed_color[1]));
    out.conic = vec3(c_a, c_b.x);
    out.opacity = c_b.y;
    out.center = (splat.xy * vec2(0.5, -0.5) + vec2(0.5, 0.5)) * camera.viewport;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let center_dist = in.center - in.position.xy;
    let power = -0.5 * (in.conic.x * pow(center_dist.x, 2.0)
                        + in.conic.z * pow(center_dist.y, 2.0))
                        - in.conic.y * center_dist.x * center_dist.y;

    if (power > 0.0) {
        return vec4(0);
    }

    let alpha = min(0.99, in.opacity * exp(power));

    return in.color * alpha;
}