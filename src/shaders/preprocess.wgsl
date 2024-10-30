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
    focal: vec2<f32>,
    fov: vec2<f32>,
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
    //TODO: store information for 2D splat rendering
    screen_pos: vec3<f32>,
    radius: f32,
};

//TODO: bind your data here

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read> gaussians : array<Gaussian>;

@group(1) @binding(0) var<storage, read_write> splats: array<Splat>;
@group(1) @binding(1) var<uniform> gs_multiplier: f32;

@group(2) @binding(0) var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1) var<storage, read_write> sort_depths : array<f32>;
@group(2) @binding(2) var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3) var<storage, read_write> sort_dispatch: DispatchIndirect;

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

fn get_rot_matrix(rot: array<u32,2>) -> mat3x3<f32> {
    let rot_xy = unpack2x16float(rot[0]);
    let rot_zr = unpack2x16float(rot[1]);
    let rot_vec4 = vec4<f32>(rot_xy.x, rot_xy.y, rot_zr.x, rot_zr.y);
    
    // Normalize quaternion
    let q = rot_vec4; // no need to normalize ? 
    let r = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;

    // Convert quaternion to rotation matrix
    return mat3x3<f32>(
        1.0 - 2.0 * (y * y + z * z),  2.0 * (x * y - z * r),      2.0 * (x * z + y * r),
        2.0 * (x * y + z * r),        1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * r),
        2.0 * (x * z - y * r),        2.0 * (y * z + x * r),      1.0 - 2.0 * (x * x + y * y)
    );
}

fn get_scale_matrix(scale: array<u32,2>, local_gs_multiplier: f32) -> mat3x3<f32> {
    let scale_xy = unpack2x16float(scale[0]);
    let scale_z_ = unpack2x16float(scale[1]);
    let scale_vec3 = vec3<f32>(scale_xy.x, scale_xy.y, scale_z_.x);
    return mat3x3<f32>(
        scale_vec3.x * local_gs_multiplier, 0.0, 0.0,
        0.0, scale_vec3.y * local_gs_multiplier, 0.0,
        0.0, 0.0, scale_vec3.z * local_gs_multiplier
    );
}

fn compute_cov_3d(
    R: mat3x3<f32>,
    S: mat3x3<f32>
) -> array<f32, 6> {
    let cov_3d_mat = R * S * transpose(S) * transpose(R);
    return array<f32, 6>(
        cov_3d_mat[0][0], cov_3d_mat[0][1], cov_3d_mat[0][2],
        cov_3d_mat[1][1], cov_3d_mat[1][2], cov_3d_mat[2][2]
    );
}

fn compute_cov_2d(
    pos: vec3<f32>,
    focal: vec2<f32>,
    viewport: vec2<f32>,
    view_matrix: mat4x4<f32>,
    cov_3d: array<f32, 6>
) -> vec3<f32> {
    // Transform point to view space
    let t = (view_matrix * vec4<f32>(pos, 1.0)).xyz;
    
    // Compute tangent of FoV
    let tan_fovx = tan(camera.fov.x * 0.5);
    let tan_fovy = tan(camera.fov.y * 0.5);
    
    // Clamp position to viewing cone
    let limx = 1.3 * tan_fovx;
    let limy = 1.3 * tan_fovy;
    let txtz = t.x / t.z;
    let tytz = t.y / t.z;
    let transformed = vec3<f32>(
        min(limx, max(-limx, txtz)) * t.z,
        min(limy, max(-limy, tytz)) * t.z,
        t.z
    );
    
    // Compute Jacobian of perspective projection
    let J = mat3x3<f32>(
        focal.x / transformed.z, 0.0, -(focal.x * transformed.x) / (transformed.z * transformed.z),
        0.0, focal.y / transformed.z, -(focal.y * transformed.y) / (transformed.z * transformed.z),
        0.0, 0.0, 0.0
    );
    
    // Extract view rotation matrix
    let W = mat3x3<f32>(
        view_matrix[0].xyz,
        view_matrix[1].xyz,
        view_matrix[2].xyz
    );
    
    // Compute transformation matrix
    let T = W * J;
    let VrK = mat3x3<f32>(
        cov_3d[0], cov_3d[1], cov_3d[2],
        cov_3d[1], cov_3d[3], cov_3d[4],
        cov_3d[2], cov_3d[4], cov_3d[5]
    );
    
    // Project covariance to 2D
    let cov = transpose(T) * VrK * T;
    
    // Apply low-pass filter (minimum pixel size)
    let cov_2d = vec3<f32>(
        cov[0][0] + 0.3,
        cov[0][1],
        cov[1][1] + 0.3
    );
    
    // Return upper triangular part (xx, xy, yy)
    return cov_2d;
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    //TODO: set up pipeline as described in instruction
    if (idx >= arrayLength(&gaussians)) {
        return;
    }
    if (idx == 0u)
    {
        atomicStore(&sort_dispatch.dispatch_x, 0u);
        sort_dispatch.dispatch_y = 1u;
        sort_dispatch.dispatch_z = 1u;
    }
    // compute screen (ndc) position of the gaussian, not pixel position, so range is [-1, 1]
    let pos_xy = unpack2x16float(gaussians[idx].pos_opacity[0]); // First u32 contains x,y as f16
    let pos_za = unpack2x16float(gaussians[idx].pos_opacity[1]); // Second u32 contains z,opacity as f16
    let global_pos = vec3<f32>(pos_xy.x, pos_xy.y, pos_za.x);
    let unhomogenized_screen_pos = camera.proj * camera.view * vec4<f32>(global_pos, 1.0);
    let homogenized_screen_pos = unhomogenized_screen_pos.xyz / unhomogenized_screen_pos.w;
    let pixel_pos = vec2<f32>(homogenized_screen_pos.x * 0.5 + 0.5, 
                              homogenized_screen_pos.y * 0.5 + 0.5); // uv center
                              
    // check if the quad is valid
    if (homogenized_screen_pos.x <= -1.2 || homogenized_screen_pos.x >= 1.2 ||
        homogenized_screen_pos.y <= -1.2 || homogenized_screen_pos.y >= 1.2) {
        return;
    }
    // store pos
    // compute 3d covariance matrix
    let R = get_rot_matrix(gaussians[idx].rot);
    let S = get_scale_matrix(gaussians[idx].scale, gs_multiplier);
    let cov_3d: array<f32, 6> = compute_cov_3d(R, S);
    // compute 2d covariance matrix
    let cov_2d: vec3<f32> = compute_cov_2d(global_pos, camera.focal, camera.viewport, camera.view, cov_3d);
    // compute conic
    let det = cov_2d.x * cov_2d.z - cov_2d.y * cov_2d.y;
    if (det == 0.0) {
        return;
    }
    let dev_inv = 1.0 / det;
    let conic = vec3<f32>(cov_2d.z, -cov_2d.y, cov_2d.x) * dev_inv;
    // find quad's radius
    let mid = 0.5 * (cov_2d.x + cov_2d.z);
    let lambda1 = mid + sqrt(max(0.1, mid * mid - det));
    let lambda2 = mid - sqrt(max(0.1, mid * mid - det));
    let radius = ceil(3.0 * sqrt(max(lambda1, lambda2)));
    // find uv

    // find clamped quad bounds

    // store pos, radius & uv
    var this_splat_index = idx; // should be sort_infos.keys_size
    atomicAdd(&sort_infos.keys_size, 1u);
    splats[this_splat_index].screen_pos = homogenized_screen_pos;
    splats[this_splat_index].radius = radius / 100.0;
    // compute color


    // if (this_splat_index % keys_per_dispatch == 0u) {
    //     atomicAdd(&sort_dispatch.dispatch_x, 1u);
    // }

    // append indices and depths
    sort_indices[this_splat_index] = this_splat_index;
    sort_depths[this_splat_index] = homogenized_screen_pos.z;

    // prepare for radix sort
    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
    let current_dispatches = (this_splat_index + 1u + keys_per_dispatch - 1u) / keys_per_dispatch;
    
    // update dispatch_x if count needs more dispatches
    let current_dispatch_x = atomicLoad(&sort_dispatch.dispatch_x);
    if (current_dispatches > current_dispatch_x) {
        atomicStore(&sort_dispatch.dispatch_x, current_dispatches);
    }
}