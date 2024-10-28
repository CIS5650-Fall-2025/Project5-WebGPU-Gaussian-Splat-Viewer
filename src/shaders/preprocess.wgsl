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
    xy: vec2f,
    size: vec2f,
};

//TODO: bind your data here
@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<uniform> render_settings: RenderSettings;

@group(1) @binding(0)
var<storage,read> gaussians : array<Gaussian>;

@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

@group(3) @binding(0)
var<storage, read_write> splats: array<Splat>;

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

    if (idx >= arrayLength(&gaussians)) {
        return;
    }

    let gaussian = gaussians[idx];

    let a = unpack2x16float(gaussian.pos_opacity[0]);
    let b = unpack2x16float(gaussian.pos_opacity[1]);
    let pos = vec4<f32>(a.x, a.y, b.x, 1.);
    // TODO(rahul): add sigmoid here
    let opacity = b.y;

    let view_space_pos = camera.view * pos;
    let clip_space_pos = camera.proj * view_space_pos;
    let screen_space_pos = clip_space_pos.xy / clip_space_pos.w;

    if (any(abs(screen_space_pos.xy) > vec2(1.2)) || view_space_pos.z < 0.0) {
        return;
    }

    let sort_index = atomicAdd(&sort_infos.keys_size, 1u);

    splats[sort_index].xy = screen_space_pos.xy;
    splats[sort_index].size = vec2(0.01, 0.01) * render_settings.gaussian_scaling;

    // TODO(rahul): remove
    let sort_depth = sort_depths[0];
    let sort_idx = sort_indices[0];
    let dispatch_z = sort_dispatch.dispatch_z;


    // let r_a = unpack2x16float(gaussian.rot[0]);
    // let r_b = unpack2x16float(gaussian.rot[1]);
    // let r = vec4(r_a, r_b);

    // let R = mat3x3f(
    //     1.f - 2.f * (r.z * r.z + r.w * r.w), 2.f * (r.y * r.z - r.x * r.w), 2.f * (r.y * r.w + r.x * r.z),
    //     2.f * (r.y * r.z + r.x * r.w), 1.f - 2.f * (r.y * r.y + r.w * r.w), 2.f * (r.z * r.w - r.x * r.y),
    //     2.f * (r.y * r.w - r.x * r.z), 2.f * (r.z * r.w + r.x * r.y), 1.f - 2.f * (r.y * r.y + r.z * r.z)
    // );

    // let s_a = unpack2x16float(gaussian.scale[0]);
    // let s_b = unpack2x16float(gaussian.scale[1]);
    // let s = exp(vec3(s_a, s_b.x));

    // let S = mat3x3f(
    //     s.x, 0, 0,
    //     0, s.y, 0,
    //     0, 0, s.z
    // ) * render_settings.gaussian_scaling;

    // let M = S * R;
    // let Sigma = transpose(M) * M;

    // // cov2d

    // var t = view_space_pos.xyz;
    // let lim = 1.3 * tan(camera.focal);
    // let ttz = t.xy / lim;
    // t = vec3(clamp(ttz, -lim, lim) * t.z, t.z);

    // let J = mat3x3f(
    //     camera.focal.x / t.z, 0.0, -(camera.focal.x * t.x) / (t.z * t.z),
    //     0.0, camera.focal.y / t.z, -(camera.focal.y * t.y) / (t.z * t.z),
    //     0, 0, 0
    // );

    // let W = transpose(mat3x3f(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz));

    // let T = W * J;

    // let Vrk = mat3x3f(
    //     Sigma[0][0], Sigma[0][1], Sigma[0][2],
    //     Sigma[0][1], Sigma[1][1], Sigma[1][2],
    //     Sigma[0][2], Sigma[1][2], Sigma[2][2]
    // );

    // var cov = transpose(T) * transpose(Vrk) * T;
    // cov[0][0] += 0.3;
    // cov[1][1] += 0.3;
    
    // // TODO(rahul): measure perf of using determinant
    // // vs manually calculating it
    // let det = determinant(cov);
    // let det_inv = 1.0 / det;
    // let conic = vec3(cov[1][1], -cov[0][1], cov[0][0]) * det_inv;

    // let mid = 0.5 * (cov[0][0] + cov[1][1]);
    // let lambda1 = mid + sqrt(max(0.1, mid * mid - det));
    // let lambda2 = mid - sqrt(max(0.1, mid * mid - det));
    // let radius = ceil(3.0 * sqrt(max(lambda1, lambda2)));

    // let sort_index = atomicAdd(&sort_infos.keys_size, 1u);
    // splats[sort_index].xy = screen_space_pos;
    // splats[sort_index].size = vec2(radius) / camera.viewport;

    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 
    // if (sort_index % keys_per_dispatch == 0) {
    //     atomicAdd(&sort_dispatch.dispatch_x, 1u);
    // }
}