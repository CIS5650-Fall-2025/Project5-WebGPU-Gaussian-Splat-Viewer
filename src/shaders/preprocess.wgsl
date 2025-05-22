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
override shDegree: u32;

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

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    //TODO: store information for 2D splat rendering
    xy: u32,
    widthHeight: u32,
    packedColor: array<u32, 2>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0)
var<storage, read> gaussians: array<Gaussian>;

//TODO: bind your data here
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
var<uniform> scaling: f32;
@group(3) @binding(2)
var<storage, read> colors: array<u32>; 
//TODO: bind your data here

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    //TODO: access your binded sh_coeff, see load.ts for how it is stored
    let idx = splat_idx * 24 + c_idx % 2 + c_idx / 2 * 3;
    let color_ab = unpack2x16float(colors[idx]);
    let color_cd = unpack2x16float(colors[idx + 1]);

    if (c_idx % 2 == 0) {
        return vec3f(color_ab.x, color_ab.y, color_cd.x);
    } else {
        return vec3f(color_ab.y, color_cd.x, color_cd.y);
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

    // early exit
    if (idx >= arrayLength(&gaussians)) {
        return;
    }

    // fetch Gaussian data
    let vert = gaussians[idx];
    let xy = unpack2x16float(vert.pos_opacity[0]);
    let za = unpack2x16float(vert.pos_opacity[1]);

    let posWorld = vec4f(xy.x, xy.y, za.x, 1.0);
    let opacity = 1.0f / (1.0f + exp(-za.y));

    // transform world position to view to clip to NDC
    let viewPos: vec4f = camera.view * posWorld;
    let clipPos: vec4f = camera.proj * viewPos;
    let ndcPos: vec3f  = clipPos.xyz / clipPos.w;

    // frustum‐cull with a 1.2× margin in XY, full [0,1] in Z
    if (ndcPos.x < -1.2 || ndcPos.x > 1.2 ||
        ndcPos.y < -1.2 || ndcPos.y > 1.2 ||
        ndcPos.z <  0.0 || ndcPos.z >  1.0) {
        return;
    }

    // unpack rotation quaternion and scale vector
    let rot0  = unpack2x16float(vert.rot[0]);
    let rot1  = unpack2x16float(vert.rot[1]);
    let rot   = vec4f(rot0.x, rot0.y, rot1.x, rot1.y);
    let r     = rot.x;
    let x     = rot.y;
    let y     = rot.z;
    let z     = rot.w;

    let sc0   = unpack2x16float(vert.scale[0]);
    let sc1   = unpack2x16float(vert.scale[1]);
    let scale = exp(vec3f(sc0.x, sc0.y, sc1.x));

    let R = mat3x3f(
        vec3f(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - r * z), 2.0 * (x * z + r * y)),
        vec3f(2.0 * (x * y + r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - r * x)),
        vec3f(2.0 * (x * z - r * y), 2.0 * (y * z + r * x), 1.0 - 2.0 * (x * x + y * y))
    );
    
    let sx = scale.x * scaling;
    let sy = scale.y * scaling;
    let sz = scale.x * scaling;
    let S = mat3x3f(
        vec3f(sx, 0.0, 0.0),
        vec3f(0.0, sy, 0.0),
        vec3f(0.0, 0.0, sz)
    );
    let M = S * R;
    let Sigma = transpose(M) * M;

    var cov3D: array<f32,6>;
    cov3D[0] = Sigma[0][0];
    cov3D[1] = Sigma[0][1];
    cov3D[2] = Sigma[0][2];
    cov3D[3] = Sigma[1][1];
    cov3D[4] = Sigma[1][2];
    cov3D[5] = Sigma[2][2];

    var t = (camera.view * posWorld).xyz;
    let invZ = 1.0 / t.z;
    let limx = 1.3 * (camera.viewport.x / (2.0 * camera.focal.x));
    let limy = 1.3 * (camera.viewport.y / (2.0 * camera.focal.y));
    t.x = clamp(t.x * invZ, -limx, limx) * t.z;
    t.y = clamp(t.y * invZ, -limy, limy) * t.z;

    // Jacobian of the projection
    let J = mat3x3f(
        vec3f(camera.focal.x * invZ,       0.0, -camera.focal.x * t.x * invZ * invZ),
        vec3f(0.0,                         camera.focal.y * invZ, -camera.focal.y * t.y * invZ * invZ),
        vec3f(0.0,                         0.0, 0.0)
    );

    let W = transpose(mat3x3f(
        camera.view[0].xyz,
        camera.view[1].xyz,
        camera.view[2].xyz
    ));

    let T = W * J;

    let Vrk = mat3x3f(
        vec3f(cov3D[0], cov3D[1], cov3D[2]),
        vec3f(cov3D[1], cov3D[3], cov3D[4]),
        vec3f(cov3D[2], cov3D[4], cov3D[5])
    );

    var Cov2D = transpose(T) * transpose(Vrk) * T;
    Cov2D[0][0] += 0.3;
    Cov2D[1][1] += 0.3;
    let cxx = Cov2D[0][0];
    let cyy = Cov2D[1][1];
    let cxy = Cov2D[0][1];

    let det = (cxx * cyy - cxy * cxy);

    if (det < 0.000001f) { return; }

    let det_inv = 1.0f / det;
    let mid = 0.5f * (cxx + cyy);
    let lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
    let lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
    let radius = ceil(3.0f * sqrt(max(lambda1, lambda2)));

    let keys_per_dispatch = workgroupSize * sortKeyPerThread;  
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys

    // reserve a slot in the sorted‐list buffer
    let splatIndex: u32 = atomicAdd(&sortInfos.keys_size, 1u);

    if (splatIndex % keys_per_dispatch == 0u) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }

    // pack and store NDC.xy into 2×16-bit floats
    let xy: u32 = pack2x16float(ndcPos.xy);
    splats[splatIndex].xy = xy;
    splats[splatIndex].widthHeight = pack2x16float(vec2f(0.005f, 0.005f) * scaling);
    let direction = normalize(posWorld.xyz - camera.view_inv[3].xyz);
    let color = computeColorFromSH(direction, idx, shDegree);

    splats[splatIndex].packed_color[0] = pack2x16float(color.rg);
    splats[splatIndex].packed_color[1] = pack2x16float(vec2f(color.b, 1.0f));
    sort_depths[splatIndex] = bitcast<u32>(100.0 - view_pos.z);
    sort_indices[splatIndex] = splatIndex;
}