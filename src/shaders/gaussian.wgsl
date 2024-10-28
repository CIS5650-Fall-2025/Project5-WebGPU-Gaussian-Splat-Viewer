struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) opacity: f32,
    @location(2) conic: vec3<f32>,
    @location(3) splatCenter: vec2<f32>,
};

struct Splat {
    packedPos: u32,
    packedSize: u32,
    packedColor: array<u32,2>,
    packedConicAndOpacity: array<u32,2>
};

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

@group(0) @binding(0) var<storage, read> splats: array<Splat>;
@group(0) @binding(1) var<storage, read> sort_indices : array<u32>;
@group(0) @binding(2) var<uniform> camera: CameraUniforms;

@vertex
fn vs_main(
    @builtin(instance_index) instanceIdx: u32,
    @builtin(vertex_index) vertIdx: u32
) -> VertexOutput {
    var out: VertexOutput;

    let splatIdx = sort_indices[instanceIdx];
    let splat = splats[splatIdx];

    // Unpack splat data
    let pos = unpack2x16float(splat.packedPos);
    let size = unpack2x16float(splat.packedSize);
    let color = vec4f(unpack2x16float(splat.packedColor[0]), unpack2x16float(splat.packedColor[1]));
    let conic = vec3f(unpack2x16float(splat.packedConicAndOpacity[0]), unpack2x16float(splat.packedConicAndOpacity[1]).x);
    let opacity = unpack2x16float(splat.packedConicAndOpacity[1]).y;

    // Draw splat as a quad (each vertex is the splat position + width/height offset)
    // Works because splat position was projected to NDC space in the preprocess step.    
    let quadVerts = array<vec2f, 6>(
        vec2f(pos.x - size.x, pos.y + size.y),
        vec2f(pos.x - size.x, pos.y - size.y),
        vec2f(pos.x + size.x, pos.y - size.y),
        vec2f(pos.x + size.x, pos.y - size.y),
        vec2f(pos.x + size.x, pos.y + size.y),
        vec2f(pos.x - size.x, pos.y + size.y)
    );

    out.position = vec4f(quadVerts[vertIdx], 0.0, 1.0);
    out.color = color;
    out.opacity = opacity;
    out.conic = conic;
    out.splatCenter = (pos.xy * vec2f(0.5, -0.5) + vec2f(0.5, 0.5)) * camera.viewport; // Store this in pixel space for easy comparison to the fragment position in the frag shader.
    return out;
}

@fragment
fn fs_main(
    in: VertexOutput
) -> @location(0) vec4<f32> {
    // https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu#L332-L352
    let distToCenter = in.splatCenter - in.position.xy;
    let power = -0.5 * (in.conic.x * pow(distToCenter.x, 2.0) 
                      + in.conic.z * pow(distToCenter.y, 2.0)) 
                      - in.conic.y * distToCenter.x * distToCenter.y;

    if (power > 0.0) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }

    let alpha = min(0.99f, in.opacity * exp(power));
    return in.color * alpha;
}