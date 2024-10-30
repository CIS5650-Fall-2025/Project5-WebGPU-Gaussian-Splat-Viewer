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
};

struct SortInfos {
    keys_size: atomic<u32>,  // instance_count in DrawIndirect
    //data below is for info inside radix sort 
    padded_size: u32, 
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
};

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
};

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>,
};

struct Splat {
    //TODO: store information for 2D splat rendering
    pos_opacity: array<u32,2>,
    color_rg_ba: array<u32,2>,
    inv_covar_2d: array<u32,2>,
    radius: u32,
};

//TODO: bind your data here

@group(0) @binding(0)
var<storage, read> gaussians : array<Gaussian>;

@group(0) @binding(1)
var<storage, read_write> splats : array<Splat>;

@group(0) @binding(2)
var<uniform> camera: CameraUniforms;

@group(0) @binding(3)
var<uniform> render_settings: RenderSettings;

@group(0) @binding(4)
var<storage, read> colors: array<u32>;

@group(1) @binding(0)
var<storage, read_write> sort_infos: SortInfos;

@group(1) @binding(1)
var<storage, read_write> sort_depths: array<u32>;

@group(1) @binding(2)
var<storage, read_write> sort_indices: array<u32>;

@group(1) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    //TODO: access your binded sh_coeff, see load.ts for how it is stored
    let max_num_coeffs = u32(16);
    let color_output_offset = splat_idx * max_num_coeffs * 3;
    let normal_idx = color_output_offset + c_idx * 3;

    //it is packed in 16, above would be for 32, so we gotta do some math
    if normal_idx % 2 == 1 {
        //null then r
        let clr_ind = (normal_idx-1) / 2;
        let packed_nr = colors[clr_ind];
        let packed_gb = colors[clr_ind + 1];
        let nr = unpack2x16float(packed_nr);
        let gb = unpack2x16float(packed_gb);
        return vec3<f32>(nr.y, gb.x, gb.y);
    } else {
        let clr_ind = normal_idx / 2;
        let packed_rg = colors[clr_ind];
        let packed_bn = colors[clr_ind + 1];
        let rg = unpack2x16float(packed_rg);
        let bn = unpack2x16float(packed_bn);
        return vec3<f32>(rg.x, rg.y, bn.x);
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

    if (idx >= arrayLength(&gaussians)) {
        return;
    }

    let gaussian = gaussians[idx];

    let pos_xy = unpack2x16float(gaussian.pos_opacity[0]);
    let pos_zo = unpack2x16float(gaussian.pos_opacity[1]);

    let position = vec4<f32>(pos_xy.x, pos_xy.y, pos_zo.x, 1.0);
    let opacity = 1.0f / (1.0f + exp(-pos_zo.y));
    //let opacity = pos_zo.y;

    let view_pos = camera.view * position;
    let proj_pose = camera.proj * view_pos;
    let pos_ndc = proj_pose.xyz / proj_pose.w;

    //culling
    if (pos_ndc.x < -1.2 || pos_ndc.x > 1.2 || pos_ndc.y < -1.2 || pos_ndc.y > 1.2 || view_pos.z <= 0.0 || view_pos.z > 100.0) {
        return;
    }

    let rx = unpack2x16float(gaussian.rot[0]);
    let yz = unpack2x16float(gaussian.rot[1]);
    let r = rx.x;
    let x = rx.y;
    let y = yz.x;
    let z = yz.y;

    
    
    let rot_mat = transpose(mat3x3<f32>(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    ));

    let scale_factors_xy = unpack2x16float(gaussian.scale[0]);
    let scale_x = exp(scale_factors_xy.x);
    let scale_y = exp(scale_factors_xy.y);
    let scale_z = exp(unpack2x16float(gaussian.scale[1]).x);

    let gs = render_settings.gaussian_scaling;
    // Construct the covar matrix in world coord frame
    let s_mat = mat3x3<f32>(
        scale_x * gs, 0.0, 0.0,
        0.0, scale_y * gs, 0.0,
        0.0, 0.0, scale_z * gs
    );
    let covar_mat_3d = rot_mat * s_mat * s_mat * transpose(rot_mat);

    let gauss_pos_view =  (camera.view * position).xyz;
    let x_p = gauss_pos_view.x;
    let y_p = gauss_pos_view.y;
    let z_p = gauss_pos_view.z;

    let jacob = mat3x3<f32>(
        camera.focal.x / z_p, 0.0, -(x_p * camera.focal.x) / (z_p * z_p),
        0.0, camera.focal.y / z_p, -(y_p * camera.focal.y) / (z_p * z_p),
        0.0, 0.0, 0.0
    );

    //copy over the W matrix, I wish there was a better way to do this with a view/slice.
    let W = mat3x3<f32>(camera.view[0][0], camera.view[1][0], camera.view[2][0],
                        camera.view[0][1], camera.view[1][1], camera.view[2][1],
                        camera.view[0][2], camera.view[1][2], camera.view[2][2]);
    
    let T_mat = W * jacob;

    var covar_mat_2d = transpose(T_mat) * covar_mat_3d * T_mat;

    //0.3 seems enormous for an epsilon value, maybe i'm misunderstanding
    covar_mat_2d[0][0] += 0.3; //put it back to 0.3 bc couldn't see bike spokes
    covar_mat_2d[1][1] += 0.3;


    let det = covar_mat_2d[0][0] * covar_mat_2d[1][1] - covar_mat_2d[0][1] * covar_mat_2d[1][0];
    let mid = 0.5 * (covar_mat_2d[0][0]+ covar_mat_2d[1][1]);
    let lambda1 = mid + sqrt(max(0.1, mid * mid - det));
    let lambda2 = mid - sqrt(max(0.1, mid * mid - det));

    let direction = normalize(position.xyz + camera.view[3].xyz);

    let covar_2d_inv = mat2x2<f32>(
        covar_mat_2d[1][1] * (1.0f / det), -covar_mat_2d[0][1] * (1.0f / det),
        -covar_mat_2d[1][0] * (1.0f / det), covar_mat_2d[0][0] * (1.0f / det)
    );

    let radius = ceil(3.0 * sqrt(max(lambda1, lambda2)));
    let rad_2d = vec2<f32>((2 * radius)/camera.viewport.x, (2 * radius)/camera.viewport.y);
    let color = computeColorFromSH(direction, idx, u32(render_settings.sh_deg));

    let sort_index = atomicAdd(&sort_infos.keys_size, 1);

    splats[sort_index].pos_opacity[0] = pack2x16float(pos_ndc.xy);
    splats[sort_index].pos_opacity[1] = pack2x16float(vec2<f32>(pos_ndc.z, opacity));
    splats[sort_index].color_rg_ba[0] = pack2x16float(color.xy);
    splats[sort_index].color_rg_ba[1] = pack2x16float(vec2<f32>(color.z, 1.0));
    splats[sort_index].inv_covar_2d[0] = pack2x16float(covar_2d_inv[0].xy);
    splats[sort_index].inv_covar_2d[1] = pack2x16float(covar_2d_inv[1].xy);
    splats[sort_index].radius = pack2x16float(rad_2d);
    //Have to make sure that the sort depth is positive, so that bitcasted sort is the same
    //farthest away sorted to first because we want to render back to front
    sort_depths[sort_index] = bitcast<u32>(101.0 - view_pos.z);
    sort_indices[sort_index] = sort_index;

    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
    if (sort_index % keys_per_dispatch == 0) {
        atomicAdd(&sort_dispatch.dispatch_x, 1);
    }
}