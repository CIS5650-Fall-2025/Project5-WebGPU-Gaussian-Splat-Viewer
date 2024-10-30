
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
    color: array<u32,2>,
    radius_depths_pos: array<u32,2>,
    conic: array<u32,2>
};

// struct Splat {
//     //TODO: store information for 2D splat rendering
//     // Data is packed, need to unpack
//     position_and_size: array<u32,2>,
//     color_data: array<u32,2>,
//     conic_and_opacity: array<u32,2>,
// };
//TODO: bind your data here
 
@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<storage,read> gaussians: array<Gaussian>;
@group(0) @binding(2)
var<storage,read_write> splats: array<Splat>;
@group(0) @binding(3)
var<uniform> settings: RenderSettings;
@group(0) @binding(4)
var<storage,read> color_data: array<u32>;

@group(1) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(1) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(1) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(1) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

/// reads the ith sh coef from the storage buffer 
// fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
//     //TODO: access your binded sh_coeff, see load.ts for how it is stored
//     let output_offset = splat_idx * 16 * 3;
//     let order_offset = c_idx * 3;
//     let index = (output_offset + order_offset)/2;
//     var coeff_1 = vec2f(0.);
//     var coeff_2 = vec2f(0.);
//     var result = vec3f(0.);
//     //if odd
//     coeff_1 = unpack2x16float((sh_buffer[output_offset + order_offset]));
//     coeff_2 = unpack2x16float((sh_buffer[output_offset + order_offset + 1]));
//     if(index % 2 == 0)
//     {
//         result = vec3f(coeff_1.x,coeff_1.y,coeff_2.x);
//     }
//     else
//     {
//         result = vec3f(coeff_1.y,coeff_2.x,coeff_2.y);
//     }
//     return result;
// }

fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    //TODO: access your binded sh_coeff, see load.ts for how it is stored
    let base = splat_idx * 24 + (c_idx / 2) * 3 + c_idx % 2;
    let rg = unpack2x16float(color_data[base]);
    let ba = unpack2x16float(color_data[base + 1]);
    if (c_idx % 2 == 0) {
        return vec3f(rg.x, rg.y, ba.x);
    } else {
        return vec3f(rg.y, ba.x, ba.y);
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


fn Compute3DCovariance(gaussian_scaling : f32, gaussian_id:u32)->array<f32,6>
{
    //  R S S' R'
    let rot_rx = unpack2x16float(gaussians[gaussian_id].rot[0]);
    let rot_yz = unpack2x16float(gaussians[gaussian_id].rot[1]);
    let r = rot_rx.x;
    let x = rot_rx.y;
    let y = rot_yz.x;
    let z = rot_yz.y; 
    let R = mat3x3f(
    1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
    2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
    2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );

    let scale_xy = unpack2x16float(gaussians[gaussian_id].scale[0]);
    let scale_zw = unpack2x16float(gaussians[gaussian_id].scale[1]);
    let S = mat3x3f(gaussian_scaling * exp(scale_xy.x),0.0,0.0,
                    0.0,gaussian_scaling * exp(scale_xy.y),0,
                    0.0,0.0,gaussian_scaling * exp(scale_zw.x));

    let M = S*R;
    let Sigma = (transpose(M)*M);

    let cov3D = array<f32,6>((Sigma[0][0]),
                            (Sigma[0][1]),
                            (Sigma[0][2]),
                            (Sigma[1][1]),
                            (Sigma[1][2]),
                            (Sigma[2][2]));

    return cov3D;

}

// 2D conic, maximum radius, and maximum quad size in NDC 
fn Compute2DCovariance(cov3D:array<f32,6>,gaussian_id:u32)->vec3f
{

    let a = unpack2x16float(gaussians[gaussian_id].pos_opacity[0]);
    let b = unpack2x16float(gaussians[gaussian_id].pos_opacity[1]);
    let pos = vec4f(a.x, a.y, b.x, 1.);
    var t = camera.view*pos;

    let limx = 0.65f * camera.viewport.x / camera.focal.x;
    let limy = 0.65f * camera.viewport.y / camera.focal.y;
    let txtz = t.x / t.z;
    let tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    let focal_x = camera.focal.x;
    let focal_y = camera.focal.y;
    let J = mat3x3f(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0.0, 0.0, 0.0);

        let W = transpose(mat3x3f(
        camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz
    ));

    let T = W * J;
    let Vrk = mat3x3f(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);
    let cov_mat = (transpose(T) * transpose(Vrk) * T);
    let cov = vec3f(cov_mat[0][0] + 0.3f,cov_mat[0][1] ,cov_mat[1][1]+ 0.3f);
    
    return cov;

}


fn ndc2Px(v:vec2f, S:vec2f)->vec2<f32>
{
	return ((v + vec2f(1.0)) * S - 1.0) * 0.5;
}


@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    
    // Early exit if index is out of bounds
    if (idx >= arrayLength(&gaussians)) {
        return;
    }

    // Load gaussian data once to avoid multiple memory accesses
    let gaussian = gaussians[idx];
    let pos_a = unpack2x16float(gaussian.pos_opacity[0]);//pos
    let pos_b = unpack2x16float(gaussian.pos_opacity[1]);//opacity
    let pos = vec4f(pos_a.x, pos_a.y, pos_b.x, 1.);

    // Transform position first
    let view_pos = camera.view * pos;
    let depths = view_pos.z;
    let proj_pos = camera.proj * view_pos;
    let ndc_pos = proj_pos.xy / proj_pos.w;

    // Compute covariance only if point is in front of camera
    if (depths <= 0.0) {
        return;
    }

    // Compute 3D covariance
    let cov3D = Compute3DCovariance(settings.gaussian_scaling, idx);
    let cov = Compute2DCovariance(cov3D, idx);

    // Compute conic safely
    let det = cov.x * cov.z - cov.y * cov.y;

    let det_inv = select(1.0 / det, 0.0, abs(det) < 1e-6);
    let conic = vec3f(
        cov.z * det_inv,
        -cov.y * det_inv,
        cov.x * det_inv
    );

    // Compute radius with bounds
    let mid = 0.5 * (cov.x + cov.z);
    let discriminant = max(0.1, mid * mid - det);
    let lambda1 = mid + discriminant;
    let lambda2 = mid - discriminant;
    let radius = ceil(3.0f * sqrt(max(lambda1, lambda2)));// Fixed radius instead of dynamic calculation to prevent potential issues

    // Compute color
    let direction = normalize(pos.xyz - camera.view_inv[3].xyz);
    let color = computeColorFromSH(direction, idx, u32(settings.sh_deg));

    // Write results atomically
    let s_idx = atomicAdd(&sort_infos.keys_size, 1u);
    
    // Pack and write data
    splats[s_idx].color[0] = pack2x16float(color.xy);
    splats[s_idx].color[1] = pack2x16float(vec2f(color.z, 1.0));
    splats[s_idx].conic[0] = pack2x16float(conic.xy);
    splats[s_idx].conic[1] = pack2x16float(vec2f(conic.z, 1.0f / (1.0f + exp(-pos_b.y))));
    splats[s_idx].radius_depths_pos[0] = pack2x16float(vec2f(radius, radius)/camera.viewport);
    splats[s_idx].radius_depths_pos[1] = pack2x16float(ndc_pos);

    // Sort handling
    let keys_per_dispatch = workgroupSize * sortKeyPerThread;
    sort_depths[s_idx] = bitcast<u32>(depths);
    sort_indices[s_idx] = s_idx;

    // Atomic operations for dispatch indirect
    
    if (s_idx % keys_per_dispatch == 0u) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}
