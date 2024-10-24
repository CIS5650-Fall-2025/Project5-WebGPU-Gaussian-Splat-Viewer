const SH_C0: f32 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f32,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

override workgroupSize: u32;
override sortKeyPerThread: u32;

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,  // instance_count in DrawIndirect
    //data below is for info inside radix sort 
    padded_size: u32, 
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct RenderSettings {
    gaussian_scaling: f32,
    sh_deg: f32,
}

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    
    // declare a packed variable for the position and size
    packed_x_y_w_h: array<u32,2>,
};

// declare the uniform buffer for the camera
@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

// declare the storage buffer for the gaussians
@group(1) @binding(0)
var<storage, read> gaussians: array<Gaussian>;

@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

// declare the storage buffer for the splats
@group(3) @binding(0)
var<storage, read_write> splats: array<Splat>;

// declare the uniform buffer for the render settings
@group(3) @binding(1)
var<uniform> render_settings: RenderSettings;

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    //TODO: access your binded sh_coeff, see load.ts for how it is stored
    return vec3<f32>(0.0);
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return  max(vec3<f32>(0.), result);
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    
    // exit the loop if the index is out of bound
    if (idx >= arrayLength(&gaussians)) {
        return;
    }
    
    // acquire the current gaussian data
    let gaussian_data = gaussians[idx];
    
    // unpack the position and opacity data
    let pos_opacity_x_y = unpack2x16float(gaussian_data.pos_opacity[0]);
    let pos_opacity_z_w = unpack2x16float(gaussian_data.pos_opacity[1]);
    let position = vec4f(pos_opacity_x_y.x, pos_opacity_x_y.y, pos_opacity_z_w.x, 1.0f);
    let opacity = pos_opacity_z_w.y;
    
    // compute the clip-space position
    let clip_space_position = camera.proj * camera.view * position;
    
    // compute the screen-space position
    let screen_space_position = clip_space_position.xy / clip_space_position.w;

    // compute the view space depth
    let view_space_depth = (camera.view * position).z;
    
    // perform frustum culling
    if (screen_space_position.x < -1.2f || screen_space_position.x > 1.2f ||
        screen_space_position.y < -1.2f || screen_space_position.y > 1.2f ||
        view_space_depth < 0.0f) {
        return;
    }
    
    // unpack the rotation
    let rotation_x_y = unpack2x16float(gaussian_data.rot[0]);
    let rotation_z_w = unpack2x16float(gaussian_data.rot[1]);
    let rotation = vec4f(
        rotation_x_y.x,
        rotation_x_y.y,
        rotation_z_w.x,
        rotation_z_w.y
    );
    let r = rotation.x;
    let x = rotation.y;
    let y = rotation.z;
    let z = rotation.w;
    
    // compute R matrix
    let R = mat3x3f(
        1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y - r * z), 2.0f * (x * z + r * y),
        2.0f * (x * y + r * z), 1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z - r * x),
        2.0f * (x * z - r * y), 2.0f * (y * z + r * x), 1.0f - 2.0f * (x * x + y * y)
    );
    
    // unpack the scale
    let scale_x_y = unpack2x16float(gaussian_data.scale[0]);
    let scale_z_w = unpack2x16float(gaussian_data.scale[1]);
    let scale = exp(vec3f(
        scale_x_y.x,
        scale_x_y.y,
        scale_z_w.x
    ));
    
    // compute the S matrix
    let S = mat3x3f(
        scale.x * render_settings.gaussian_scaling, 0.0f, 0.0f,
        0.0f, scale.y * render_settings.gaussian_scaling, 0.0f,
        0.0f, 0.0f, scale.z * render_settings.gaussian_scaling
    );
    
    // compute the M matrix
    let M = S * R;
    
    // compute the 3D covariance matrix
    let three_dimensional_covariance_matrix = transpose(M) * M;
    
    // compute the 3D covariance
    let three_dimensional_covariance = array<f32, 6>(
        three_dimensional_covariance_matrix[0][0],
        three_dimensional_covariance_matrix[0][1],
        three_dimensional_covariance_matrix[0][2],
        three_dimensional_covariance_matrix[1][1],
        three_dimensional_covariance_matrix[1][2],
        three_dimensional_covariance_matrix[2][2],
    );
    
    // compute the t vector
    var t = (camera.view * position).xyz;
    let limx = 0.65f * camera.viewport.x / camera.focal.x;
    let limy = 0.65f * camera.viewport.y / camera.focal.y;
    let txtz = t.x / t.z;
    let tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;
    
    // compute the J matrix
    let J = mat3x3f(
        camera.focal.x / t.z, 0.0f, -(camera.focal.x * t.x) / (t.z * t.z),
        0.0f, camera.focal.y / t.z, -(camera.focal.y * t.y) / (t.z * t.z),
        0.0f, 0.0f, 0.0f
    );
    
    // compute the W matrix
    let W = transpose(mat3x3f(
        camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz
    ));
    
    // compute the T matrix
    let T = W * J;
    
    // compute the V matrix
    let V = mat3x3f(
        three_dimensional_covariance[0],
        three_dimensional_covariance[1],
        three_dimensional_covariance[2],
        three_dimensional_covariance[1],
        three_dimensional_covariance[3],
        three_dimensional_covariance[4],
        three_dimensional_covariance[2],
        three_dimensional_covariance[4],
        three_dimensional_covariance[5]
    );
    
    // compute the 2D covariance matrix
    var two_dimensional_covariance_matrix = transpose(T) * transpose(V) * T;
    two_dimensional_covariance_matrix[0][0] += 0.3f;
    two_dimensional_covariance_matrix[1][1] += 0.3f;
    
    // compute the 2D covariance
    let two_dimensional_covariance = vec3f(
        two_dimensional_covariance_matrix[0][0],
        two_dimensional_covariance_matrix[0][1],
        two_dimensional_covariance_matrix[1][1]
    );
    
    // compute the determinant
    var determinant = two_dimensional_covariance.x * two_dimensional_covariance.z;
    determinant -= two_dimensional_covariance.y * two_dimensional_covariance.y;
    if (determinant == 0.0f) {
        return;
    }
    
    // compute the radius
    let mid = (two_dimensional_covariance.x + two_dimensional_covariance.z) * 0.5f;
    let lambda1 = mid + sqrt(max(0.1f, mid * mid - determinant));
    let lambda2 = mid - sqrt(max(0.1f, mid * mid - determinant));
    let radius = ceil(3.0f * sqrt(max(lambda1, lambda2)));
    
    // compute the size
    let size = vec2f(radius, radius) / camera.viewport;
    
    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
    
    // acquire some data from each bound resource for testing
    let view = camera.view;
    let pos_opacity = gaussians[idx].pos_opacity;
    let passes = sort_infos.passes;
    let sort_depth = sort_depths[0];
    let sort_index = sort_indices[0];
    let dispatch_z = sort_dispatch.dispatch_z;
    
    // atomically increment the key size and acquire the index
    let index = atomicAdd(&sort_infos.keys_size, 1u);
    
    // pack the position and size
    let packed_x_y = pack2x16float(screen_space_position);
    let packed_w_h = pack2x16float(size);
    
    // update the splat data
    splats[index].packed_x_y_w_h[0] = packed_x_y;
    splats[index].packed_x_y_w_h[1] = packed_w_h;
    
    // update the sorting data
    sort_depths[index] = bitcast<u32>(100.0f - view_space_depth);
    sort_indices[index] = index;
    
    // increase the dispatch group count everything index exceeds the work group size
    if (index % keys_per_dispatch == 0) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}
