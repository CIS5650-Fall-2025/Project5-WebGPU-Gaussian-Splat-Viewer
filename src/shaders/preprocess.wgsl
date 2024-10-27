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
    mean_xy: u32,
    radii: u32,
    conic_xy: u32,
    conic_z: u32,
    rgb_rg: u32,
    rgb_b_opacity: u32,
};

//TODO: bind your data here
@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<uniform> settings: RenderSettings;
@group(0) @binding(2)
var<storage, read> sh_coefficients: array<array<u32, 24>>;
@group(0) @binding(3)
var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(4)
var<storage, read_write> splats: array<Splat>;

@group(1) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(1) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(1) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(1) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    //TODO: access your binded sh_coeff, see load.ts for how it is stored
    return vec3<f32>(
        unpack2x16float(sh_coefs[splat_idx][(3u*c_idx)*0.5])[(3u*c_idx) % 2u], 
        unpack2x16float(sh_coefs[splat_idx][(3u*c_idx+1u)*0.5])[(3u*c_idx+1u) % 2u], 
        unpack2x16float(sh_coefs[splat_idx][(3u*c_idx+2u)*0.5])[(3u*c_idx+2u) % 2u]
    );
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

// Normalize so that the normal is a unit vector.
fn normalizePlane(plane: vec4<f32>) -> vec4<f32> {
    let length = length(plane.xyz);
    return plane / length;
}

// Gribb-Hartmann. www.gamedevs.org/uploads/fast-extraction-viewing-frustum-planes-from-world-view-projection-matrix.pdf.
fn extractFrustumPlanes(view_proj: mat4x4<f32>) -> array<vec4<f32>, 6> {
    var planes: array<vec4<f32>, 6>;

    // Left.
    planes[0] = normalizePlane(view_proj[3] + view_proj[0]);
    // Right.
    planes[1] = normalizePlane(view_proj[3] - view_proj[0]);
    // Bottom.
    planes[2] = normalizePlane(view_proj[3] + view_proj[1]);
    // Top.
    planes[3] = normalizePlane(view_proj[3] - view_proj[1]);
    // Near.
    planes[4] = normalizePlane(view_proj[3] + view_proj[2]);
    // Far.
    planes[5] = normalizePlane(view_proj[3] - view_proj[2]);

    // In clip space.
    return planes;
}

fn frustumCull(pos: vec4<f32>, scale: vec3<f32>, view_proj: mat4x4<f32>) -> bool {
    var enlarged_scale = abs(scale) * 1.1;
    let pos_in_view_space = (camera.view * pos).xyz;
    let min_bounds = pos_in_view_space.xyz - enlarged_scale;
    let max_bounds = pos_in_view_space.xyz + enlarged_scale;

    let min_clip = view_proj * vec4<f32>(min_bounds, 1.0);
    let max_clip = view_proj * vec4<f32>(max_bounds, 1.0);

    let frustum_planes = extractFrustumPlanes(view_proj);

    for (var i = 0u; i < 6u; i = i + 1u) {
        let plane = frustum_planes[i];

        let p = vec3<f32>(
            select(min_clip.x, max_clip.x, plane.x > 0.0),
            select(min_clip.y, max_clip.y, plane.y > 0.0),
            select(min_clip.z, max_clip.z, plane.z > 0.0)
        );

        if (dot(plane.xyz, p) + plane.w < 0.0) {
            // Cull.
            return false;
        }
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
        1.0 - 2.0 * (y2 + z2), 2.0 * (xy - wz),       2.0 * (xz + wy),
        2.0 * (xy + wz),       1.0 - 2.0 * (x2 + z2), 2.0 * (yz - wx),
        2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (x2 + y2)
    );
}

// Based on original author's CUDA implementation.
// github.com/graphdeco-inria/diff-gaussian-rasterization/blob/59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d/cuda_rasterizer/forward.cu#L118
fn computeCov3D(gaussian: Gaussian) -> array<f32, 6> {
    var scale = vec3<f32>(unpack2x16float(gaussian.scale[0]).xy, unpack2x16float(gaussian.scale[1]).x);
    scale = exp(scale);
    let S = mat3x3f(
        scale.x, 0.0f, 0.0f,
        0.0f, scale.y, 0.0f,
        0.0f, 0.0f, scale.z
    ) * settings.gaussian_scaling;

    let rot_ax = unpack2x16float(gaussian.rot[0]);
    let rot_yz = unpack2x16float(gaussian.rot[1]);
    let quaternion = vec4<f32>(rot_ax.y, rot_yz.x, rot_yz.y, rot_ax.x);
    let R = quaternionToMatrix(quaternion);

    let M = S * R;
    
    let Sigma = transpose(M) * M;

    let cov3D = array<f32, 6>(
        Sigma[0][0],
        Sigma[0][1],
        Sigma[0][2],
        Sigma[1][1],
        Sigma[1][2],
        Sigma[2][2],
    );

    return cov3D;
}

// Based on original author's CUDA implementation.
// github.com/graphdeco-inria/diff-gaussian-rasterization/blob/59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d/cuda_rasterizer/forward.cu#L74
fn computeCov2D(cov3D: array<f32, 6>, mean: vec4f) -> vec3f {
    var t = (camera.view * mean).xyz;

    let limx = 0.65f * camera.viewport.x / camera.focal.x;
    let limy = 0.65f * camera.viewport.y / camera.focal.y;
    let txtz = t.x / t.z;
    let tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    let J = mat3x3f(
        camera.focal.x / t.z, 0.0f,                 -(camera.focal.x * t.x) / (t.z * t.z),
        0.0f,                 camera.focal.y / t.z, -(camera.focal.y * t.y) / (t.z * t.z),
        0.0f,                 0.0f,                 0.0f
    );

    let W = mat3x3f(
        camera.view[0].x, camera.view[1].x, camera.view[2].x,
        camera.view[0].y, camera.view[1].y, camera.view[2].y,
        camera.view[0].z, camera.view[1].z, camera.view[2].z
    );

    let T = W * J;

    let Vrk = mat3x3f(
        cov3D[0], cov3D[1], cov3D[2],
        cov3D[1], cov3D[3], cov3D[4],
        cov3D[2], cov3D[4], cov3D[5]
    );

    var cov2d = transpose(T) * transpose(Vrk) * T;
    cov2d[0][0] += 0.3f;
    cov2d[1][1] += 0.3f;

    return vec3f(cov2d[0][0], cov2d[0][1], cov2d[1][1]);
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    //TODO: set up pipeline as described in instruction

    let gaussian = gaussians[idx];
    
    let pos_xy = unpack2x16float(gaussian.pos_opacity[0]);
    let pos_z_opacity = unpack2x16float(gaussian.pos_opacity[1]);
    let mean = vec4f(pos_xy.x, pos_xy.y, pos_z_opacity.x, 1.0f);
    
    let mean_clip = camera.proj * camera.view * mean;
    let mean_screen = mean_clip.xy / mean_clip.w;
    if (frustumCull(mean_clip)) {
        return;
    }

    // Covariance, radii, and conic computation is based on the paper author's
    // CUDA implementation.
    // github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu.

    let cov = computeCov2D(computeCov3D(gaussian), mean);

    var det = cov.x * cov.z - cov.y * cov.y;
    if (det == 0.0) {
        return;
    }
    let det_inv = 1.0 / det;

    let conic = vec3f(cov.z*det_inv, -cov.y*det_inv, cov.x*det_inv);
    
    let mid = 0.5 * (cov.x + cov.z);
    let lambda1 = mid + sqrt(max(0.1, mid*mid - det));
    let lambda2 = mid - sqrt(max(0.1, mid*mid - det));
    let radius = ceil(3.0 * sqrt(max(lambda1, lambda2)));

    let cam_pos_world = camera.view_inv[3].xyz;
    let rgb = computeColorFromSH(
        normalize(mean.xyz - cam_pos_world),
        idx, 
        u32(settings.sh_deg)
    );

    let key = atomicAdd(&sort_infos.keys_size, 1u);
    let z_far = camera.proj[3][2] / (1.0 - camera.proj[2][2]);
    sort_depths[key] = bitcast<u32>(z_far - (camera.view * mean).z);
    sort_indices[key] = key;

    splats[key].mean_xy = pack2x16float(mean_screen);
    splats[key].radii = pack2x16float(vec2f(radius, radius) / camera.viewport);
    splats[key].conic_xy = pack2x16float(conic.xy);
    splats[key].conic_z = pack2x16float(vec2f(conic.z, 0.0));
    splats[key].rgb_rg = pack2x16float(rgb.rg);
    splats[key].rgb_b_opacity = pack2x16float(vec2f(rgb.b, pos_z_opacity.y));

    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
    if (key % keys_per_dispatch == 0) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}