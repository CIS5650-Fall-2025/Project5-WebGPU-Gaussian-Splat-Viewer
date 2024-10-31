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

    radii_depths_pos:array<u32,2>,
    conic_opacity:array<u32,2>,
    color_tiles_touched:array<u32,2>
};

//TODO: bind your data here

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0)
var<storage,read> gaussians : array<Gaussian>;
@group(1) @binding(1)
var<uniform> settings:RenderSettings;
@group(1) @binding(2) 
var<storage,read> sh_buffer : array<u32>;
@group(1) @binding(3)
var<storage,read_write> splats:array<Splat>;

@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    //TODO: access your binded sh_coeff, see load.ts for how it is stored
    let base = splat_idx * 24 + (c_idx / 2) * 3 + c_idx % 2;
    let rg = unpack2x16float(sh_buffer[base]);
    let ba = unpack2x16float(sh_buffer[base + 1]);
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


fn computeCov3D(gaussian_scaling:f32,idx:u32) -> array<f32,6>
{
	// Create scaling matrix
	
    let scalingxy = unpack2x16float(gaussians[idx].scale[0]);
    let scalingzw = unpack2x16float(gaussians[idx].scale[1]);
	let S = mat3x3f(gaussian_scaling * exp(scalingxy.x),0.,0.,
	 0.,gaussian_scaling * exp(scalingxy.y),0.,
	 0.,0.,gaussian_scaling * exp(scalingzw.x));

	// Normalize quaternion to get valid rotation
    let rotxy = unpack2x16float(gaussians[idx].rot[0]);
    let rotzw = unpack2x16float(gaussians[idx].rot[1]);
	let r = rotxy.x;
	let x = rotxy.y;
	let y = rotzw.x;
	let z = rotzw.y;

	// Compute rotation matrix from quaternion
	let R = mat3x3f(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	let M = S * R;

	// Compute 3D world covariance matrix Sigma
	let Sigma = transpose(M) * M;

	// Covariance is symmetric, only store upper right
    var cov3D =array<f32,6>(
	    Sigma[0][0],
	    Sigma[0][1],
	    Sigma[0][2],
	    Sigma[1][1],
	    Sigma[1][2],
	    Sigma[2][2]);

    return cov3D;
}

fn computeCov2D(worldpos: vec3f, cov3D:array<f32,6>)->vec3f
{
    //view space point
	var t =  camera.view * vec4f(worldpos,1.);

    let limx = 0.65f * camera.viewport.x / camera.focal.x;
    let limy = 0.65f * camera.viewport.y / camera.focal.y;

	let txtz = t.x / t.z;
	let tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	let J = mat3x3f(
		camera.focal.x / t.z, 0.0f, -(camera.focal.x * t.x) / (t.z * t.z),
		0.0f, camera.focal.y / t.z, -(camera.focal.y * t.y) / (t.z * t.z),
		0, 0, 0);

	let W = transpose(mat3x3f(camera.view[0].xyz,
                    camera.view[1].xyz,
                    camera.view[2].xyz));

	let T = W * J;

	let Vrk = mat3x3f(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	var cov = transpose(T) * transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return vec3f(cov[0][0] ,cov[0][1],cov[1][1]);
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    //TODO: set up pipeline as described in instruction
    let pos_xy = unpack2x16float(gaussians[idx].pos_opacity[0]);//world pos xy
    let pos_z_opacity = unpack2x16float(gaussians[idx].pos_opacity[1]);//world pos z and opacity

    let worldpos = vec4f(pos_xy,pos_z_opacity.x,1.);

    //compute depth for depth based view frustum culling
    let depth = (camera.view * worldpos).z;
    if(depth <= 0.1) // if depth less than near plane(hard coded to 0.1)
    {
        return;
    }

    //compute 3D, 2D covariance and inv 2D cov as conic
    let cov3D = computeCov3D(settings.gaussian_scaling,idx);
    let cov = computeCov2D(worldpos.xyz, cov3D);
    let det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
    {
		return;
    }
	let det_inv = 1.f / det;
	let conic = vec3f( cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv );

    let mid = 0.5f * (cov.x + cov.z);
	let lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	let lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	let my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));

    let projpos = camera.proj * camera.view * worldpos;
    let ndc = projpos.xy/projpos.w;

    let direction = normalize(worldpos.xyz - camera.view_inv[3].xyz);
    let color = computeColorFromSH(direction,idx,u32(settings.sh_deg));
    //set info for sorter
    let keysize = atomicAdd(&sort_infos.keys_size,1u);

    splats[keysize].radii_depths_pos[0] = pack2x16float(vec2f(f32(my_radius),f32(depth)));
    splats[keysize].radii_depths_pos[1] = pack2x16float(ndc.xy);

    splats[keysize].conic_opacity[0] = pack2x16float(vec2f(conic.xy));
    splats[keysize].conic_opacity[1] = pack2x16float(vec2f(conic.z,1.0f / (1.0f + exp(-pos_z_opacity.y))));

    let grid = vec3f(20.,20.,1.);
    let rect_min = vec2f(
		min(grid.x, max(0., ((projpos.x - my_radius) / 20.))),
		min(grid.y, max(0., ((projpos.y - my_radius) / 20.)))
    );
	let rect_max = vec2f(
		min(grid.x, max(0., ((projpos.x + my_radius + 20. - 1) / 20.))),
		min(grid.y, max(0., ((projpos.y + my_radius + 20. - 1) / 20.)))
    );

    splats[keysize].color_tiles_touched[0] = pack2x16float(color.xy);
    splats[keysize].color_tiles_touched[1] = pack2x16float(vec2f(color.z,1.));
    
    sort_depths[keysize] =  bitcast<u32>(100.0f - depth);
    sort_indices[keysize] = keysize;

    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 

    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
    if(keysize%keys_per_dispatch == 0)
    {
        atomicAdd(&sort_dispatch.dispatch_x,1u);
    }


}
