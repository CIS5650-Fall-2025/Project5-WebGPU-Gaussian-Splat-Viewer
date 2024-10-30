struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
    @location(0) center:  vec2f,
    @location(1) color:   vec4f,
    @location(2) conic:   vec3f,
    @location(3) opacity: f32
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
    //TODO: information defined in preprocess compute shader
    packed_pos:  u32,
    packed_size: u32,
    packed_color: array<u32, 2>,
    packed_conic_opacity: array<u32, 2>
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read> splat_buffer: array<Splat>;
@group(0) @binding(2) var<storage, read> sorting_info: array<u32>;

@vertex
fn vs_main(
    @builtin(instance_index) instance_idx: u32,
    @builtin(vertex_index) vertex_idx: u32
) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass 
    var out: VertexOutput;

    let splat_idx = sorting_info[instance_idx];
    let splat = splat_buffer[splat_idx];

    let pos = unpack2x16float(splat.packed_pos);
    let size = unpack2x16float(splat.packed_size);
    let x = pos.x;
    let y = pos.y;
    let w = size.x;
    let h = size.y;

    let vertices = array<vec2f, 6>(
        vec2f(x - w, y + h),
        vec2f(x - w, y - h),
        vec2f(x + w, y - h),
        vec2f(x + w, y - h),
        vec2f(x + w, y + h),
        vec2f(x - w, y + h)
    );
    out.position = vec4f(vertices[vertex_idx], 0.0, 1.0);

    out.color = vec4f(unpack2x16float(splat.packed_color[0]),
                      unpack2x16float(splat.packed_color[1]));
    out.conic = vec3f(unpack2x16float(splat.packed_conic_opacity[0]),
                      unpack2x16float(splat.packed_conic_opacity[1]).x);
    out.opacity = unpack2x16float(splat.packed_conic_opacity[1]).y;
    out.center = (0.5 + pos * vec2f(0.5, -0.5)) * camera.viewport;
   
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu#L263
    let d = in.center - in.position.xy;
    let p = -0.5 * (in.conic.x * d.x * d.x + in.conic.z * d.y * d.y)
                - in.conic.y * d.x * d.y;

    if (p > 0.0) { return vec4f(0.0, 0.0, 0.0, 0.0); }

    let alpha = min(0.99f, in.opacity * exp(p));
    return in.color * alpha;
}