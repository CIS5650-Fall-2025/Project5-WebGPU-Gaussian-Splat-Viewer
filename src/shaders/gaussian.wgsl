struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) conic_matrix_0: vec3<f32>,
    @location(1) conic_matrix_1: vec3<f32>,
    @location(2) conic_matrix_2: vec3<f32>,
    @location(3) color: vec4<f32>
};

struct Splat {
    pos_ndc: vec3<f32>,
    size_ndc: vec2<f32>,
    conic_matrix: mat3x3<f32>,
    color: vec4<f32>
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
var<storage, read> splats: array<Splat>;

@vertex
fn vs_main(
) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass 
    var out: VertexOutput;
    let splat = splats[vertex_index / 6u];

    let quad_positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0)
    );

    let vertex_pos_ndc = vec2<f32>(
        quad_positions[vertex_index].x * splat.size_ndc.x + splat.pos_ndc.x,
        quad_positions[vertex_index].y * splat.size_ndc.y + splat.pos_ndc.y
    );

    out.position = vec4<f32>(vertex_pos_ndc, splat.pos_ndc.z, 1.0);

    out.conic_matrix_0 = splat.conic_matrix[0];
    out.conic_matrix_1 = splat.conic_matrix[1];
    out.conic_matrix_2 = splat.conic_matrix[2];
    out.color = splat.color;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var pos_ndc = 2.0 * (input.position.xy / camera.viewport) - vec2(1.0, 1.0);
    pos_ndc.y = -pos_ndc.y;

    let conic_matrix = mat3x3<f32>(
        in.conic_matrix_0,
        in.conic_matrix_1,
        in.conic_matrix_2
    );

    let distance_from_center = dot(pos_ndc, conic_matrix * pos_ndc);

    if (distance_from_center > 1.0) {
        discard;
    }

    let decay = exp(-distance_from_center);

    return vec4<f32>(in.color.rgb, in.color.a * decay);
}