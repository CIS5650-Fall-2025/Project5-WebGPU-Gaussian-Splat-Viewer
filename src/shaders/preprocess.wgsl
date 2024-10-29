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
    //TODO: store information for 2D splat rendering
    // Data is packed, need to unpack
    position_and_size: array<u32,2>,
    color_data: array<u32,2>,
    conic_and_opacity: array<u32,2>,
};

//TODO: bind your data here
@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(2)
var<storage, read_write> splat_elements: array<Splat>;
@group(0) @binding(3)
var<uniform> settings: RenderSettings;
@group(0) @binding(4)
var<storage, read> color_data: array<u32>;

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
    let base = splat_idx * 24 + (c_idx / 2) * 3 + c_idx % 2;
    if (c_idx % 2 == 0) {
        let rg = unpack2x16float(color_data[base]);
        let ba = unpack2x16float(color_data[base + 1]);
        return vec3f(rg.x, rg.y, ba.x);
    } else {
        let rg = unpack2x16float(color_data[base]);
        let ba = unpack2x16float(color_data[base + 1]);
        return vec3f(rg.y, rg.x, ba.y);
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
    let idx = gid.x;
    //TODO: set up pipeline as described in instruction
    // Reference - https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu
    if (idx >= arrayLength(&gaussians)) {
        return;
    }
    
    let current_element = gaussians[idx];
    
    let xy = unpack2x16float(current_element.pos_opacity[0]);
    let opacity = unpack2x16float(current_element.pos_opacity[1]);
    let position = vec4f(xy.x, xy.y, opacity.x, 1.0f);
    
    let clip_pos = camera.proj * camera.view * position;
    let ndc = clip_pos.xy / clip_pos.w;
    let view_depth = (camera.view * position).z;
    
    if (abs(ndc.x) > 1.0f ||
        abs(ndc.y) > 1.0f ||
        view_depth < 0.0f) {
        return;
    }
    // 3D Covariance
    let S1 = unpack2x16float(current_element.scale[0]);
    let S2 = unpack2x16float(current_element.scale[1]);
    let R1 = unpack2x16float(current_element.rot[0]);
    let R2 = unpack2x16float(current_element.rot[1]);
    let S = mat3x3f(
        // 1
        exp(S1.x) * settings.gaussian_scaling, 0.0f, 0.0f,
        // 2
        0.0f, exp(S1.y) * settings.gaussian_scaling, 0.0f,
        // 3
        0.0f, 0.0f, exp(S2.x) * settings.gaussian_scaling
    );
    let R = mat3x3f(
        // 1
        1.0f - 2.0f * (R2.x * R2.x + R2.y * R2.y),
        2.0f * (R1.y * R2.x - R1.x * R2.y),
        2.0f * (R1.y * R2.y + R1.x * R2.x),
        // 2
        2.0f * (R1.y * R2.x + R1.x * R2.y),
        1.0f - 2.0f * (R1.y * R1.y + R2.y * R2.y),
        2.0f * (R2.x * R2.y - R1.x * R1.y),
        // 3
        2.0f * (R1.y * R2.y - R1.x * R2.x),
        2.0f * (R2.x * R2.y + R1.x * R1.y),
        1.0f - 2.0f * (R1.y * R1.y + R2.x * R2.x)
    );
    
    let M = S * R;
    let sigma  = transpose(M) * M;
    let cov3D = array<f32, 6>(
        sigma [0][0],
        sigma [0][1],
        sigma [0][2],
        sigma [1][1],
        sigma [1][2],
        sigma [2][2],
    );
    
    // 2D
    var t = (camera.view * position).xyz;
    let limx = 0.65f * camera.viewport.x / camera.focal.x;
    let limy = 0.65f * camera.viewport.y / camera.focal.y;
    let txtz = t.x / t.z;
    let tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;
    
    let J = mat3x3f(
        camera.focal.x / t.z, 0.0f, -(camera.focal.x * t.x) / (t.z * t.z),
        0.0f, camera.focal.y / t.z, -(camera.focal.y * t.y) / (t.z * t.z),
        0.0f, 0.0f, 0.0f
    );
    let W = transpose(mat3x3f(
        camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz
    ));
    
    let T = W * J;
    
    let Vrk = mat3x3f(
        // 1
        cov3D[0], cov3D[1], cov3D[2],
        // 2
        cov3D[1], cov3D[3], cov3D[4],
        // 3
        cov3D[2], cov3D[4], cov3D[5]
    );
    
    var cov = transpose(T) * transpose(Vrk) * T;
    cov[0][0] += 0.3f;
    cov[1][1] += 0.3f;
    
    let cov2d = vec3f(
        cov[0][0],
        cov[0][1],
        cov[1][1]
    );
    
    var det  = (cov2d.x * cov2d.z) - (cov2d.y * cov2d.y);
    if (det  == 0.0f) {
        return;
    }
    var det_inv = 1.f / det;
    let conic = vec3f(
        cov2d.z * det_inv, -cov2d.y * det_inv, cov2d.x * det_inv,
    );
    
    let mid = 0.5f * (cov2d.x + cov2d.z);
    let lambda1 = mid + sqrt(max(0.1f, mid * mid - det ));
    let lambda2 = mid - sqrt(max(0.1f, mid * mid - det ));
    let my_radius  = ceil(3.0f * sqrt(max(lambda1, lambda2)));
    let size = vec2f(my_radius , my_radius ) / camera.viewport;
    
    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
    let splat_index = atomicAdd(&sort_infos.keys_size, 1u);
    let direction = normalize(position.xyz - camera.view_inv[3].xyz);
    let color = computeColorFromSH(direction, idx, u32(settings.sh_deg));
    
    splat_elements[splat_index].position_and_size[0] = pack2x16float(ndc);
    splat_elements[splat_index].position_and_size[1] = pack2x16float(size);
    splat_elements[splat_index].color_data[0] = pack2x16float(color.rg);
    splat_elements[splat_index].color_data[1] = pack2x16float(vec2f(color.b, 1.0f));
    splat_elements[splat_index].conic_and_opacity[0] = pack2x16float(conic.xy);
    splat_elements[splat_index].conic_and_opacity[1] = pack2x16float(vec2f(conic.z, 1.0f / (1.0f + exp(-opacity.y))));
    
    sort_depths[splat_index] = bitcast<u32>(100.0f - view_depth);
    sort_indices[splat_index] = splat_index;
    if (splat_index % keys_per_dispatch == 0) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}
