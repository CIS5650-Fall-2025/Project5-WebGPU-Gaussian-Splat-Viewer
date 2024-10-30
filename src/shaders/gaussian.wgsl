struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) wh: vec2f,
    @location(1) color: vec4f
};

struct Splat {
    xy: u32,
    wh: u32,
    rg: u32,
    ba: u32
};

@group(0) @binding(0)
var<storage, read> splats: array<Splat>;
@group(0) @binding(1)
var<storage, read> sort_indices : array<u32>;

@vertex
fn vs_main(
    @builtin(instance_index) instanceIndex: u32,
    @builtin(vertex_index) vertexIndex: u32
) -> VertexOutput {
    var out: VertexOutput;

    let index = sort_indices[instanceIndex];
    let splat = splats[index];
    let xy = unpack2x16float(splat.xy);
    let wh = unpack2x16float(splat.wh);

    let corners = array<vec2f, 6>(
        vec2f(xy.x - wh.x, xy.y + wh.y),
        vec2f(xy.x - wh.x, xy.y - wh.y),
        vec2f(xy.x + wh.x, xy.y - wh.y),
        vec2f(xy.x + wh.x, xy.y - wh.y),
        vec2f(xy.x + wh.x, xy.y + wh.y),
        vec2f(xy.x - wh.x, xy.y + wh.y),
    );
    let pos = vec4(corners[vertexIndex].x, corners[vertexIndex].y, 0, 1);

    let rg = unpack2x16float(splat.rg);
    let ba = unpack2x16float(splat.ba);
    let color = vec4f(rg.x, rg.y, ba.x, ba.y);

    out.position = pos;
    out.wh = wh;
    out.color = color;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    //return vec4<f32>(in.wh.x, in.wh.y, 0.0, 1.0);
    return in.color;
}