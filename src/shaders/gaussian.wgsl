struct VertexOutput {
    @builtin(position) position: vec4<f32>,

    // declare the variable for the vertex color
    @location(0) color: vec4f,
    
    // declare the variable for the conic and opacity
    @location(1) conic_opacity: vec4f,
    
    // declare the variable for the center
    @location(2) center: vec2f,
};

struct Splat {
    
    // declare a packed variable for the position and size
    packed_x_y_w_h: array<u32,2>,
    
    // declare a packed variable for the color
    packed_color: array<u32,2>,
    
    // declare a packed variable for the conic and opacity
    packed_conic_opacity: array<u32,2>,
};

// declare the struct for the camera data
struct CameraData {
    
    // declare a variable for the view matrix
    view_matrix: mat4x4f,
    
    // declare a variable for the inverse of the view matrix
    inverse_view_matrix: mat4x4f,
    
    // declare a variable for the projection matrix
    projection_matrix: mat4x4f,
    
    // declare a variable for the inverse of the projection matrix
    inverse_projection_matrix: mat4x4f,

    // declare a variable for the viewport
    viewport: vec2f,
    
    // declare a variable for the focal data
    focal: vec2f,
};

// declare the storage buffer for the splats
@group(0) @binding(0)
var<storage, read> splats: array<Splat>;

// declare the storage buffer for the sort indices
@group(0) @binding(1)
var<storage, read> sort_indices : array<u32>;

// declare the uniform buffer for the camera
@group(0) @binding(2)
var<uniform> camera: CameraData;

@vertex
fn vs_main(

    // declare the argument for the instance index
    @builtin(instance_index) global_instance_index: u32,
    
    // declare the argument for the vertex index
    @builtin(vertex_index) local_vertex_index: u32
    
) -> VertexOutput {
    
    // acquire the splat data index
    let index = sort_indices[global_instance_index];
    
    // acquire the current splat data
    let splat = splats[index];
    
    // unpack the position and size
    let unpacked_x_y = unpack2x16float(splat.packed_x_y_w_h[0]);
    let unpacked_w_h = unpack2x16float(splat.packed_x_y_w_h[1]);
    let x = unpacked_x_y.x;
    let y = unpacked_x_y.y;
    let w = unpacked_w_h.x * 2.0f;
    let h = unpacked_w_h.y * 2.0f;
    
    // create an array of positions
    let positions = array<vec2f, 6>(
        vec2f(x - w, y + h),
        vec2f(x - w, y - h),
        vec2f(x + w, y - h),
        vec2f(x + w, y - h),
        vec2f(x + w, y + h),
        vec2f(x - w, y + h),
    );
    
    // compute the vertex position
    let position = vec4(
        positions[local_vertex_index].x,
        positions[local_vertex_index].y,
        0.0f, 1.0f
    );
    
    // unpack the color
    let unpacked_r_g = unpack2x16float(splat.packed_color[0]);
    let unpacked_b_a = unpack2x16float(splat.packed_color[1]);
    let r = unpacked_r_g.x;
    let g = unpacked_r_g.y;
    let b = unpacked_b_a.x;
    let a = unpacked_b_a.y;

    // compute the vertex color
    let color = vec4(r, g, b, a);

    // unpack the conic and opacity
    let unpacked_conic_xy = unpack2x16float(splat.packed_conic_opacity[0]);
    let unpacked_conic_z_opacity = unpack2x16float(splat.packed_conic_opacity[1]);
    let conic = vec3f(unpacked_conic_xy.x, unpacked_conic_xy.y, unpacked_conic_z_opacity.x);
    let opacity = unpacked_conic_z_opacity.y;

    // compute the size
    let size = (unpacked_w_h * 0.5f + 0.5f) * camera.viewport.xy;

    // create the vertex output data
    var vertex_output: VertexOutput;
    
    // update the vertex output position
    vertex_output.position = position;
    
    // update the vertex output color
    vertex_output.color = color;
    
    // update the vertex output conic and opacity
    vertex_output.conic_opacity = vec4f(conic, opacity);
    
    // update the vertex output center
    vertex_output.center = vec2f(x, y);

    // return the vertex output data
    return vertex_output;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    
    // compute the screen-space position
    var position = (in.position.xy / camera.viewport) * 2.0f - 1.0f;
    position.y *= -1.0f;

    // compute the offset in screen-space position
    var offset = position.xy - in.center.xy;
    offset.x *= -1.0f;
    offset *= camera.viewport * 0.5f;

    // compute the power
    var power = in.conic_opacity.x * pow(offset.x, 2.0f);
    power += in.conic_opacity.z * pow(offset.y, 2.0f);
    power *= -0.5f;
    power -= in.conic_opacity.y * offset.x * offset.y;

    // return nothing if the power is greater than zero
    if (power > 0.0f) {
        return vec4f(0.0f, 0.0f, 0.0f, 0.0f);
    }
    
    // compute the alpha
    let alpha = min(0.99f, in.conic_opacity.w * exp(power));
    
    // return the output color
    return in.color * alpha;
}