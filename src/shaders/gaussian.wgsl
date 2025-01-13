struct CameraUniforms {
    view: mat4x4f,
    viewInv: mat4x4f,
    proj: mat4x4f,
    projInv: mat4x4f,
    viewport: vec2f,
    focal: vec2f
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) center: vec2f,
    @location(1) color: vec3f,
    @location(2) conic: vec3f,
    @location(3) opacity: f32
};

struct Splat {
    pos: vec2f,
    size: vec2f,
    color: vec3f,
    conic: vec3f,
    opacity: f32
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0)
var<storage, read> splats: array<Splat>;
@group(1) @binding(1)
var<storage, read> sortIndices : array<u32>;

@vertex
fn vs_main(
    @builtin(instance_index) instanceIndex: u32,
    @builtin(vertex_index) vertexIndex: u32
) -> VertexOutput {
    const vertices = array<vec2f, 4>(
        vec2f(-1, -1),
        vec2f( 1, -1),
        vec2f(-1,  1),
        vec2f( 1,  1)
    );

    let splat = splats[sortIndices[instanceIndex]];

    var out: VertexOutput;
    out.position = vec4f(splat.pos + splat.size * vertices[vertexIndex], 0.0, 1.0);
    out.center = (vec2f(0.5, -0.5) * splat.pos + 0.5) * camera.viewport;
    out.color = splat.color;
    out.opacity = splat.opacity;
    out.conic = splat.conic;
    return out;
}

@fragment
fn fs_main(
    in: VertexOutput
) -> @location(0) vec4f {
    let d = in.center - in.position.xy;
    let power = -0.5 * (in.conic.x * d.x * d.x + in.conic.z * d.y * d.y) - in.conic.y * d.x * d.y;

    if (power > 0.0) { discard; }

    let alpha = min(0.99, in.opacity * exp(power));

    // Discard if alpha is less than 1/255
    if (alpha < 0.00392156862) { discard; }

    // Premultiplied alpha
    return vec4f(in.color * alpha, alpha);
}