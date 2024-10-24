struct VertexOutput {
    @builtin(position) position: vec4<f32>,

    // declare the variable for the vertex color
    @location(0) color: vec4f,
};

struct Splat {
    
    // declare a packed variable for the position and size
    packed_x_y_w_h: array<u32,2>,
    
    // declare a packed variable for color
    packed_color: array<u32,2>,
};

// declare the storage buffer for the splats
@group(0) @binding(0)
var<storage, read> splats: array<Splat>;

// declare the storage buffer for the sort indices
@group(0) @binding(1)
var<storage, read> sort_indices : array<u32>;

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

    // create the vertex output data
    var vertex_output: VertexOutput;
    
    // update the vertex output position
    vertex_output.position = position;
    
    // update the vertex output color
    vertex_output.color = color;

    // return the vertex output data
    return vertex_output;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    
    // return the vertex color for testing
    return in.color;
}