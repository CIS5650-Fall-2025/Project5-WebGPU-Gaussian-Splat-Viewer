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
    xy: u32,
    wh: u32,
    rg: u32,
    ba: u32
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0)
var<storage, read> gaussians: array<Gaussian>;

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
var<storage, read> shs: array<u32>;

fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    let i = splat_idx * 24 + (c_idx / 2) * 3 + c_idx % 2;

    if (c_idx % 2 == 0) {
        let rg = unpack2x16float(shs[i + 0]);
        let br = unpack2x16float(shs[i + 1]);
        return vec3f(rg.x, rg.y, br.x);
    } else {
        let br = unpack2x16float(shs[i + 0]);
        let gb = unpack2x16float(shs[i + 1]);
        return vec3f(br.y, gb.x, gb.y);
    }
}

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

fn quatToRotationMatrix(rot: vec4f) -> mat3x3f {
    let r = rot.x;
    let x = rot.y;
    let y = rot.z;
    let z = rot.w;

    return mat3x3f(
        vec3f(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - r * z), 2.0 * (x * z + r * y)),
        vec3f(2.0 * (x * y + r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - r * x)),
        vec3f(2.0 * (x * z - r * y), 2.0 * (y * z + r * x), 1.0 - 2.0 * (x * x + y * y))
    );
}

fn computeCov3D(scale: vec3f, scaling: f32, rot: vec4f) -> array<f32, 6> {
    let S = mat3x3f(
        vec3f(scaling * scale.x, 0.0, 0.0),
        vec3f(0.0, scaling * scale.y, 0.0),
        vec3f(0.0, 0.0, scaling * scale.z)
    );

    let R = quatToRotationMatrix(rot);

    let M = S * R;

    let Sigma = transpose(M) * M;

    var cov3D: array<f32, 6>;
    cov3D[0] = Sigma[0][0];
    cov3D[1] = Sigma[0][1];
    cov3D[2] = Sigma[0][2];
    cov3D[3] = Sigma[1][1];
    cov3D[4] = Sigma[1][2];
    cov3D[5] = Sigma[2][2];

    return cov3D;
}

fn computeCov2D(pos: vec4f, focal_x: f32, focal_y: f32, tan_fovx: f32, tan_fovy: f32, cov3D: array<f32, 6>, view: mat4x4f) -> vec3f {
    var t = (view * pos).xyz;

    let limx = 1.3f * tan_fovx;
    let limy = 1.3f * tan_fovy;
    let txtz = t.x / t.z;
    let tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    let J = mat3x3f(
        vec3f(focal_x / t.z, 0.0, -(focal_x * t.x) / (t.z * t.z)),
        vec3f(0.0, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z)),
        vec3f(0.0, 0.0, 0.0)
    );

    let W = transpose(mat3x3f(view[0].xyz, view[1].xyz, view[2].xyz));

    let T = W * J;

    let Vrk = mat3x3f(
        vec3f(cov3D[0], cov3D[1], cov3D[2]),
        vec3f(cov3D[1], cov3D[3], cov3D[4]),
        vec3f(cov3D[2], cov3D[4], cov3D[5])
    );

    var cov = transpose(T) * transpose(Vrk) * T;

    cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;

    return vec3f(cov[0][0], cov[0][1], cov[1][1]);
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {

    let idx = gid.x;
    
    if (idx >= arrayLength(&gaussians)) {
        return;
    }
    
    let vertex = gaussians[idx];
    let a = unpack2x16float(vertex.pos_opacity[0]);
    let b = unpack2x16float(vertex.pos_opacity[1]);
    let pos_world = vec4f(a.x, a.y, b.x, 1.0f);
    let opa = b.y;
    let pos_view = camera.view * pos_world;
    let pos_clip = camera.proj * pos_view;
    let pos_ndc = pos_clip.xyz / pos_clip.w;

    if (pos_ndc.x < -1.2f || pos_ndc.x > 1.2f ||
        pos_ndc.y < -1.2f || pos_ndc.y > 1.2f ||
        pos_ndc.z < 0.00f || pos_ndc.z > 1.0f) {
        return;
    }

    let rot_a = unpack2x16float(vertex.rot[0]);
    let rot_b = unpack2x16float(vertex.rot[1]);
    let rot = vec4f(rot_a.x, rot_a.y, rot_b.x, rot_b.y);

    let sca_a = unpack2x16float(vertex.scale[0]);
    let sca_b = unpack2x16float(vertex.scale[1]);
    let scale = exp(vec3f(sca_a.x, sca_a.y, sca_b.x));
    
    let cov3D = computeCov3D(scale, scaling, rot);
    let cov = computeCov2D(
        pos_world, 
        camera.focal.x, camera.focal.y, 
        camera.viewport.x / (2.f * camera.focal.x), camera.viewport.y / (2.f * camera.focal.y),
        cov3D, camera.view);
    let det = (cov.x * cov.z - cov.y * cov.y);

    if (det < 0.000001f) {
        return;
    }

    let det_inv = 1.0f / det;
    let conic = vec3f(cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv);

    let mid = 0.5f * (cov.x + cov.z);
    let lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
    let lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
    let radius = ceil(3.f * sqrt(max(lambda1, lambda2)));

    let keys_per_dispatch = workgroupSize * sortKeyPerThread;
    let index = atomicAdd(&sort_infos.keys_size, 1u);
    if (index % keys_per_dispatch == 0) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }

    let xy = pack2x16float(pos_ndc.xy);
    let wh = pack2x16float((vec2f(radius, radius) * 2.0f / camera.viewport) * scaling);
    splats[index].xy = xy;
    splats[index].wh = wh;

    let dir = normalize(pos_world.xyz - camera.view_inv[3].xyz);
    let color = computeColorFromSH(dir, idx, shDegree);
    let rg = pack2x16float(color.rg);
    let ba = pack2x16float(vec2f(color.b, 1.0f));
    splats[index].rg = rg;
    splats[index].ba = ba;

    sort_depths[index] = bitcast<u32>(100.0f - pos_view.z);
    sort_indices[index] = index;
}