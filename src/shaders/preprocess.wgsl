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
    // 4 16-bit floats for position (x,y,z) and opacity packed as 2 u32.
    pos_opacity: array<u32,2>,
    // 4 16-bit floats for rotation (x,y,z,w) packed as 2 u32.
    rot: array<u32,2>,
    // 3 16-bit floats for scale (x,y,z) packed as 2 u32.
    scale: array<u32,2>
};

struct Splat {
    dummy: vec3<f32>,
};

//TODO: bind your data here
@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

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

fn frustumCull(pos: vec4<f32>, scale: vec3<f32>, camera: CameraUniforms) -> bool {
    let pos_in_view_space = (camera.view * pos).xyz;

    var enlarged_scale = abs(scale) * 1.1;
    let min_bounds = pos_in_view_space - enlarged_scale;
    let max_bounds = pos_in_view_space + enlarged_scale;

    let near_plane = 0.01;
    let far_plane = 100.0;

    if (max_bounds.z < near_plane || min_bounds.z > far_plane) {
        // Before or beyond near/far planes.
        return false;
    }

    // Isosceles triangle up to aabb min z.
    let left_plane = min_bounds.z * -1.0;
    // Isosceles triangle up to aabb max z.
    let right_plane = max_bounds.z * 1.0;
    if (max_bounds.x < left_plane || min_bounds.x > right_plane) {
        return false;
    }

    // Similarly.
    let bottom_plane = min_bounds.z * -1.0;
    let top_plane = max_bounds.z * 1.0;
    if (max_bounds.y < bottom_plane || min_bounds.y > top_plane) {
        return false;
    }

    return true;
}

fn quaternionToMatrix(q: vec4<f32>) -> mat3x3<f32> {
    let x2 = q.x * q.x;
    let y2 = q.y * q.y;
    let z2 = q.z * q.z;
    let xy = q.x * q.y;
    let xz = q.x * q.z;
    let yz = q.y * q.z;
    let wx = q.w * q.x;
    let wy = q.w * q.y;
    let wz = q.w * q.z;

    return mat3x3<f32>(
        1.0 - 2.0 * (y2 + z2), 2.0 * (xy - wz),      2.0 * (xz + wy),
        2.0 * (xy + wz),      1.0 - 2.0 * (x2 + z2), 2.0 * (yz - wx),
        2.0 * (xz - wy),      2.0 * (yz + wx),       1.0 - 2.0 * (x2 + y2)
    );
}

fn computeCovarianceMatrix(rotation: vec4<f32>, scale: vec3<f32>, scaling_factor: f32) -> mat3x3<f32> {
    let R = quaternionToMatrix(rotation);
    let RT = transpose(R);

    let S = mat3x3<f32>(
        vec3<f32>(scale.x, 0.0, 0.0),
        vec3<f32>(0.0, scale.y, 0.0),
        vec3<f32>(0.0, 0.0, scale.z)
    ) * scaling_factor;
    let ST = transpose(S);

    let covariance = R * S * ST * RT;
    return covariance;
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    //TODO: set up pipeline as described in instruction

    // let focal = camera.focal;

    // let gaussian = gaussians[idx];

    // let padded_size = sort_infos.padded_size;
    // let sort_depth = sort_depths[0];
    // let sort_indices = sort_indices[0];
    // let dispatch_y = sort_dispatch.dispatch_y;

    let gaussian = gaussians[idx];
    
    let pos = vec4<f32>(unpack2x16float(gaussian.pos_opacity[0]).xy, unpack2x16float(gaussian.pos_opacity[1]).x, 1.0);
    let scale = vec3<f32>(unpack2x16float(gaussian.scale[0]).xy, unpack2x16float(gaussian.scale[1]).x);

    // Quaternion.
    let rot_xy = unpack2x16float(gaussian.rot[0]);
    let rot_zw = unpack2x16float(gaussian.rot[1]);
    let rotation = vec4<f32>(rot_xy.x, rot_xy.y, rot_zw.x, rot_zw.y);

    let view_proj = camera.proj * camera.view;

    if (!frustumCull(pos, scale, view_proj)) {
        return;
    }

    let covariance = computeCovarianceMatrix(rotation, scale, 1.0);

    let pos_ndc = camera.proj * pos;
    let max_radius = length(scale) * 1.1;
    let size_ndc = vec2<f32>(max_radius, max_radius);

    atomicAdd(&sort_infos.keys_size, 1u);
    sort_depths[idx] = u32(pos_ndc.z * 1000.0);
    sort_indices[idx] = idx;

    // Dispatch handling logic
    let keys_per_dispatch = workgroupSize * sortKeyPerThread;
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys

    if (idx % keys_per_dispatch == 0u) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}