struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
};

struct Splat {
    
    // declare a packed variable for the position and size
    packed_x_y_w_h: array<u32,2>,
};

@vertex
fn vs_main(

    // declare the argument for the vertex index
    @builtin(vertex_index) local_vertex_index: u32
    
) -> VertexOutput {
    
    // create some temporary data for testing
    let x = 0.0f;
    let y = 0.0f;
    let w = 0.1f;
    let h = 0.1f;
    
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
    
    // create the vertex output data
    var vertex_output: VertexOutput;
    
    // update the vertex output position
    vertex_output.position = position;
    
    // return the vertex output data
    return vertex_output;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1.);
}