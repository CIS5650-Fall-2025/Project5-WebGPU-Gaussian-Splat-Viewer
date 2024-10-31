struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader

    @location(0) conic_opacity:vec4f,
    @location(1) color:vec4f,
    @location(2) center:vec2f
};

struct Splat {
    //TODO: information defined in preprocess compute shader
    radii_depths_pos:array<u32,2>,
    conic_opacity:array<u32,2>,
    color_tiles_touched:array<u32,2>
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
var<storage> splats:array<Splat>;
@group(0) @binding(2)
var<storage> sort_indices : array<u32>;

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertexIndex: u32,
    @builtin(instance_index) in_instanceIndex: u32,
) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass 
    var out: VertexOutput;

    let radius = unpack2x16float(splats[sort_indices[in_instanceIndex]].radii_depths_pos[0]).x;
    let ndcPos = unpack2x16float(splats[sort_indices[in_instanceIndex]].radii_depths_pos[1]);

    let pos = array<vec2f, 6>(
        vec2f((ndcPos.x - radius * 2.0f/camera.viewport.x), (ndcPos.y + radius * 2.0f/camera.viewport.y)),
        vec2f((ndcPos.x - radius * 2.0f/camera.viewport.x), (ndcPos.y - radius * 2.0f/camera.viewport.y)),
        vec2f((ndcPos.x + radius * 2.0f/camera.viewport.x), (ndcPos.y - radius * 2.0f/camera.viewport.y)),
        vec2f((ndcPos.x + radius * 2.0f/camera.viewport.x), (ndcPos.y - radius * 2.0f/camera.viewport.y)),
        vec2f((ndcPos.x + radius * 2.0f/camera.viewport.x), (ndcPos.y + radius * 2.0f/camera.viewport.y)),
        vec2f((ndcPos.x - radius * 2.0f/camera.viewport.x), (ndcPos.y + radius * 2.0f/camera.viewport.y)),
    );
    out.position = vec4f(pos[in_vertexIndex].xy,0.,1.);
    let color_xy = unpack2x16float(splats[sort_indices[in_instanceIndex]].color_tiles_touched[0]);
    let color_zw = unpack2x16float(splats[sort_indices[in_instanceIndex]].color_tiles_touched[1]);
    let conic_xy = unpack2x16float(splats[sort_indices[in_instanceIndex]].conic_opacity[0]);
    let conic_zw = unpack2x16float(splats[sort_indices[in_instanceIndex]].conic_opacity[1]);
    out.color = vec4f(color_xy,color_zw.x,1.);
    out.conic_opacity = vec4f(conic_xy,conic_zw);
    out.center = ndcPos.xy;
    return out;
}


@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {

    var center = vec2f((in.center.x + 1.)*0.5*camera.viewport.x,(1. - in.center.y)*0.5*camera.viewport.y );
    let pixelPos = in.position.xy;
    var offset = (pixelPos - center);
    offset.x *= -1;
    var power = -0.5f * (
        in.conic_opacity.x * pow(offset.x, 2.0f) + 
        in.conic_opacity.z * pow(offset.y, 2.0f)
    );
    power -= in.conic_opacity.y * offset.x * offset.y;

    if (power > 0.0f) {
        return vec4f(0.0f, 0.0f, 0.0f, 0.0f);
    }
    
    let alpha = min(0.99f, in.conic_opacity.w * exp(power));
    return in.color * alpha;

}
