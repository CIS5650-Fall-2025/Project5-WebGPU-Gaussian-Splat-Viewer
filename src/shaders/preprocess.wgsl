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

    packed_color: array<u32, 2>,
    packed_conic_opacity: array<u32, 2>
};

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
@group(3) @binding(1)
var<storage, read> sh: array<u32>;

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    // maximum 16 sh coefficients per splat, which is 16*3 = 48 f16s = 48 * 2 = 96 bytes
    // since we are encoding them as u32s, we need 24 u32s = 24 * 4 = 96 bytes.
    let parity = c_idx % 2;
    let index = splat_idx * 24 + (c_idx / 2) * 3 + parity;
    let c_a = unpack2x16float(sh[index + 0]);
    let c_b = unpack2x16float(sh[index + 1]);
    if (parity == 0) {
        return vec3(c_a, c_b.x);
    } else {
        return vec3(c_a.y, c_b);
    }
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
    let index = gid.x;

    if (index >= arrayLength(&gaussians)) {
        return;
    }

    let gaussian = gaussians[index];

    let a = unpack2x16float(gaussian.pos_opacity[0]);
    let b = unpack2x16float(gaussian.pos_opacity[1]);
    let pos = vec4<f32>(a.x, a.y, b.x, 1.);
    let opacity = 1.0 / (1.0 + exp(-b.y));

    let view_space_pos = camera.view * pos;
    let clip_space_pos = camera.proj * view_space_pos;
    let screen_space_pos = clip_space_pos.xy / clip_space_pos.w;

    if (any(abs(screen_space_pos.xy) > vec2(1.2)) || view_space_pos.z < 0.0) {
        return;
    }

    let r = vec4(unpack2x16float(gaussian.rot[0]), unpack2x16float(gaussian.rot[1]));
    let R = mat3x3f(
        1.f - 2.f * (r.z * r.z + r.w * r.w), 2.f * (r.y * r.z - r.x * r.w), 2.f * (r.y * r.w + r.x * r.z),
        2.f * (r.y * r.z + r.x * r.w), 1.f - 2.f * (r.y * r.y + r.w * r.w), 2.f * (r.z * r.w - r.x * r.y),
        2.f * (r.y * r.w - r.x * r.z), 2.f * (r.z * r.w + r.x * r.y), 1.f - 2.f * (r.y * r.y + r.z * r.z)
    );

    let s = exp(vec3(unpack2x16float(gaussian.scale[0]), unpack2x16float(gaussian.scale[1]).x));
    let S = mat3x3f(
        s.x, 0, 0,
        0, s.y, 0,
        0, 0, s.z
    ) * render_settings.gaussian_scaling;
    
    let M = S * R;
    let Sigma = transpose(M) * M;
    
    var t = view_space_pos.xyz;
    let lim = 0.65 * camera.viewport * camera.focal;
    let ttz = t.xy / t.z;
    t = vec3(clamp(ttz, -lim, lim) * t.z, t.z);

    let J = mat3x3f(
        camera.focal.x / t.z, 0.0, -(camera.focal.x * t.x) / (t.z * t.z),
        0.0, camera.focal.y / t.z, -(camera.focal.y * t.y) / (t.z * t.z),
        0, 0, 0
    );
    
    let W = transpose(mat3x3f(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz));

    let T = W * J;

    let Vrk = mat3x3f(
        Sigma[0][0], Sigma[0][1], Sigma[0][2],
        Sigma[0][1], Sigma[1][1], Sigma[1][2],
        Sigma[0][2], Sigma[1][2], Sigma[2][2]
    );
    
    var cov = transpose(T) * transpose(Vrk) * T;
    cov[0][0] += 0.3;
    cov[1][1] += 0.3;
    
    let det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
    let conic = vec3(cov[1][1], -cov[0][1], cov[0][0]) / det;
    
    let mid = 0.5 * (cov[0][0] + cov[1][1]);
    let lambda1 = mid + sqrt(max(0.1, mid * mid - det));
    let lambda2 = mid - sqrt(max(0.1, mid * mid - det));
    let radius = ceil(3.0 * sqrt(max(lambda1, lambda2)));
    
    let direction = normalize(pos.xyz - camera.view_inv[3].xyz);
    let color = computeColorFromSH(direction, index, u32(render_settings.sh_deg));

    let sort_index = atomicAdd(&sort_infos.keys_size, 1u);

    splats[sort_index].xy = screen_space_pos.xy;
    splats[sort_index].size = vec2(radius, radius) / camera.viewport;

    splats[sort_index].packed_color[0] = pack2x16float(color.rg);
    splats[sort_index].packed_color[1] = pack2x16float(vec2(color.b, 1.0));

    splats[sort_index].packed_conic_opacity[0] = pack2x16float(conic.xy);
    splats[sort_index].packed_conic_opacity[1] = pack2x16float(vec2(conic.z, opacity));

    sort_indices[sort_index] = sort_index;
    sort_depths[sort_index] = bitcast<u32>(100.0f - view_space_pos.z);

    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 
    if (sort_index % keys_per_dispatch == 0) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}