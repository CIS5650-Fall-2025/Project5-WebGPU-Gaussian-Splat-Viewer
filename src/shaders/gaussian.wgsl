struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) size: vec2<f32>,
    @location(1) color: vec4<f32>, //TODO: information passed from vertex shader to fragment shader
};

struct Splat {
    //TODO: information defined in preprocess compute shader
    xyPacked: u32,
    widthHeight: u32,
    packedColor: array<u32, 2>
};

@group(0) @binding(0)
var<storage, read_write> splats: array<Splat>;
@group(0) @binding(1)
var<storage, read> sort_indices : array<u32>;
@group(0) @binding(2)
var<uniform> camera: CameraUniforms;

@vertex
fn vs_main(
    @builtin(instance_index) instanceIndex: u32,
    @builtin(vertex_index) vertexIndex: u32
) -> VertexOutput {
    var out: VertexOutput;

    // unpack center XY from 2×16-bit floats
    let splatIdx = sort_indices[instanceIndex];
    let splat = splats[splatIdx];
    let xy = unpack2x16float(splat.xyPacked);
    let size: vec2f = unpack2x16float(splat.widthHeight);

    // apply corner offset
    let corners = array<vec2f, 6>(
        vec2f(xy.x - size.x, xy.y + size.y),
        vec2f(xy.x - size.x, xy.y - size.y),
        vec2f(xy.x + size.x, xy.y - size.y),
        vec2f(xy.x + size.x, xy.y - size.y),
        vec2f(xy.x + size.x, xy.y + size.y),
        vec2f(xy.x - size.x, xy.y + size.y)
    );
    let corner = corners[vertexIndex];
    out.position = vec4f(corner.x, corner.y, 0.0, 1.0);
    out.size = size;

    let color = unpack2x16float(splat.packedColor[0]);
    out.color = vec4f(unpack2x16float(splat.packed_color[0]),
                      unpack2x16float(splat.packed_color[1]));
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color
}