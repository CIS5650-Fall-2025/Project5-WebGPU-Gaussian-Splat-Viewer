struct CameraUniforms {
    view: mat4x4f,
    viewInv: mat4x4f,
    proj: mat4x4f,
    projInv: mat4x4f,
    viewport: vec2f,
    focal: vec2f,
    clippingPlanes: vec2f
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    // No interpolation
    @location(0) @interpolate(flat) center: u32,
    @location(1) @interpolate(flat) color_opacity: vec2u,
    @location(2) @interpolate(flat) conic: vec2u,
};

struct Splat {
    pos: u32,
    size: u32,
    color_opacity: vec2u,
    conic: vec2u
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
    let position = unpack2x16float(splat.pos);
    let size = unpack2x16float(splat.size);

    var out: VertexOutput;
    out.position = vec4f(position + size * vertices[vertexIndex], 0.0, 1.0);
    out.center = pack2x16float((vec2f(0.5, -0.5) * position + 0.5) * camera.viewport);
    out.color_opacity = splat.color_opacity;
    out.conic = splat.conic;
    return out;
}

@fragment
fn fs_main(
    in: VertexOutput
) -> @location(0) vec4f {
    let conic = vec4f(
        unpack2x16float(in.conic.x),
        unpack2x16float(in.conic.y)
    );
    let center = vec2f(unpack2x16float(in.center));
    let color_opacity = vec4f(
        unpack2x16float(in.color_opacity.x),
        unpack2x16float(in.color_opacity.y)
    );

    let d = center - in.position.xy;
    let power = 0.5 * (conic.x * d.x * d.x + conic.z * d.y * d.y) + conic.y * d.x * d.y;

    if (power < 0.0) { discard; }



    let alpha = min(0.99, color_opacity.a * exp(-power));

    // Discard if alpha is less than 1/255
    if (alpha < 0.00392156862) { discard; }

    // Premultiplied alpha
    return vec4f(color_opacity.rgb * alpha, alpha);
}