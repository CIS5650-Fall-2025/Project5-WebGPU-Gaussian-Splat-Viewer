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
    color: vec3<f32>,
    radius: f32,
    depths: f32,
    conic: vec3f,
    projpos: vec2f
};

//TODO: bind your data here
 
@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(1) @binding(0)
var<storage,read> gaussians: array<Gaussian>;
@group(1) @binding(1)
var<storage,read_write> splats: array<Splat>;
@group(1) @binding(2)
var<storage,read> sh_buffer: array<f32>;

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
    let output_offset = splat_idx * 16 * 3;
    let order_offset = c_idx * 3;
    
    return vec3<f32>(sh_buffer[output_offset +order_offset],sh_buffer[output_offset +order_offset+1],sh_buffer[output_offset +order_offset+2]);
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
    let S = mat3x3f(gaussian_scaling * scale_xy.x,0.0,0.0,
                    0.0,gaussian_scaling * scale_xy.y,0,
                    0.0,0.0,gaussian_scaling * scale_zw.x);

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
    let t = pos*camera.view;
    let focal_x = camera.focal.x;
    let focal_y = camera.focal.y;
    let J = mat4x4f(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),0.0,
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),0.0,
		0.0, 0.0, 0.0,0.0,
        0.0, 0.0, 0.0,0.0);

    let T = camera.view * J;
    let Vrk = mat4x4f(
		cov3D[0], cov3D[1], cov3D[2],0.0,
		cov3D[1], cov3D[3], cov3D[4],0.0,
		cov3D[2], cov3D[4], cov3D[5],0.0,
        0.0, 0.0, 0.0,0.0);
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
    //TODO: set up pipeline as described in instruction
// Implement view frustum culling to remove non-visible Gaussians (make bounding box to be slightly larger to keep the edge gaussians)
// Compute 3D covariance based on rotation and scale, also user inputted gaussian multipler. (see post on 1.1 section)
// Compute 2D conic, maximum radius, and maximum quad size in NDC (see post on 1.1 section)
// Using spherical harmonics coeffiecients to evaluate the color of the gaussian from particular view direction (evaluation function is provided, see post ).
// Store essential 2D gaussian data for later rasteriation pipeline
// Add key_size, indices, and depth to sorter.
    let cov3D = Compute3DCovariance(1.0,idx);
    let cov = Compute2DCovariance(cov3D,idx);

    let det = f32(cov.x * cov.z - cov.y * cov.y);
    let det_inv = f32(1.f / det);
    let conic = vec3f(cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv );
    
    let mid = f32(0.5f * (cov.x + cov.z));
    let lambda1 = f32(mid + sqrt(max(0.1f, mid * mid - det)));  
    let lambda2 = f32(mid - sqrt(max(0.1f, mid * mid - det)));
    let my_radius = ceil(3.f * sqrt(max(lambda1, lambda2))); 

    let a = unpack2x16float(gaussians[idx].pos_opacity[0]);
    let b = unpack2x16float(gaussians[idx].pos_opacity[1]);
    let pos = vec4<f32>(a.x, a.y, b.x, 1.);

    let viewprojmat = camera.proj*camera.view;
    var projPos = viewprojmat * pos;
    projPos /= projPos.w; // perspective divide
    let projPosPx = vec2f((projPos.x + 1.0)* camera.viewport.x*0.5, (1.0 - projPos.y)*camera.viewport.y*0.5);

    
    splats[idx].conic = conic;
    splats[idx].projpos = projPosPx;

    splats[idx].depths = ((camera.view * pos).z);
    splats[idx].radius = my_radius;
    splats[idx].color = computeColorFromSH(pos.xyz,idx, 2);

    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 
    sort_depths[idx] = u32(splats[idx].depths);
    sort_indices[idx] = idx;
    atomicAdd(&sort_infos.keys_size,u32(1));

    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
    let key_size = atomicLoad(&sort_infos.keys_size);
    if(key_size==keys_per_dispatch)
    {
        atomicStore(&sort_infos.keys_size,0u);
        atomicAdd(&sort_dispatch.dispatch_x,u32(1));
    }

}

