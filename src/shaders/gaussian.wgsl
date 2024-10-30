struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

 const quad_verts = array<vec2<f32>, 6>(
    vec2<f32>(-1.0f, 1.0f),  
    vec2<f32>(-1.0f, -1.0f), 
    vec2<f32>(1.0f,  -1.0f),  
    vec2<f32>(1.0f, -1.0f),  
    vec2<f32>(1.0f,  1.0f),  
    vec2<f32>(-1.0f, 1.0f),  
);

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) v_color: vec4<f32>,
    @location(1) conic: vec3<f32>,
    @location(2) opacity: f32,
    @location(3) center: vec2<f32>,
};



struct Splat {
    //TODO: information defined in preprocess compute shader
    packedposition: u32, 
    packedsize: u32,          
    packedcolor: array<u32,2>, 
    packedconic_opacity: array<u32,2>
};

@group(0) @binding(0)var<storage, read> splats: array<Splat>;
@group(0) @binding(1)var<storage, read> sorted_indices : array<u32>;
@group(0) @binding(2) var<uniform> camera: CameraUniforms;


@vertex
fn vs_main(
    @builtin(instance_index) instanceIndex: u32,
    @builtin(vertex_index) vert_idx : u32,) -> VertexOutput {
    var out: VertexOutput;

    let index = sorted_indices[instanceIndex];
    var splat = splats[index];
    let posi = unpack2x16float(splat.packedposition);
    let size = unpack2x16float(splat.packedsize);
    let conicXY = unpack2x16float(splat.packedconic_opacity[0]);
    let conicZO = unpack2x16float(splat.packedconic_opacity[1]);
    let conic = vec3<f32>(conicXY, conicZO.x);
    let op = conicZO.y;
    let colorRG = unpack2x16float(splat.packedcolor[0]);
    let colorBA = unpack2x16float(splat.packedcolor[1]);
    let color = vec4<f32>(colorRG, colorBA);

    let offset = quad_verts[vert_idx];
    let position = vec2<f32>(posi.x + offset.x * size.x, posi.y + offset.y * size.y);

    out.position = vec4<f32>(position, 0.0, 1.0);
    out.v_color = color;
    out.conic = conic;
    out.opacity = op;
    out.center = vec2<f32>(posi.xy);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var NDCpos = (in.position.xy / camera.viewport) * 2.0f - 1.0f;
    NDCpos.y = -NDCpos.y;

    //back to pixel space
    var offset = (NDCpos.xy - in.center.xy) * camera.viewport * 0.5f;
    let power = -0.5f * (in.conic.x * (-offset.x) * (-offset.x) + in.conic.z * offset.y * offset.y) - in.conic.y * (-offset.x) * offset.y;

    if (power > 0.0f){
        return vec4<f32>(0.0f);
    }
    let alpha = min(0.99f, in.opacity * exp(power));

    return in.v_color * alpha;
    //return vec4<f32>(offset,0.0, 1.0);
}