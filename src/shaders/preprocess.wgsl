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
    packedPos: u32,
    packedSize: u32,
    packedColor: array<u32,2>,
    packedConicAndOpacity: array<u32,2>
};

//TODO: bind your data here
@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<uniform> render_settings: RenderSettings;
@group(1) @binding(0)
var<storage, read> gaussians: array<Gaussian>;
@group(1) @binding(1)
var<storage, read> colors: array<u32>;

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
    let gaussian_base_index = splat_idx * 24u;

    // Compute the base index
    let base_index = gaussian_base_index + (c_idx / 2u) * 3u + c_idx % 2u;

    var result: vec3<f32>;

    if (c_idx % 2u == 0u) {
        // Even c_idx
        let color_r_g = unpack2x16float(colors[base_index + 0u]);
        let color_b_r = unpack2x16float(colors[base_index + 1u]);
        result = vec3<f32>(color_r_g.x, color_r_g.y, color_b_r.x);
    } else {
        // Odd c_idx
        let color_b_r = unpack2x16float(colors[base_index + 0u]);
        let color_g_b = unpack2x16float(colors[base_index + 1u]);
        result = vec3<f32>(color_b_r.y, color_g_b.x, color_g_b.y);
    }

    return result;
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

    // Decode position and opacity from Gaussian data
    let xy = unpack2x16float(gaussian.pos_opacity[0]);
    let zo = unpack2x16float(gaussian.pos_opacity[1]);
    let pos_world = vec3<f32>(xy.x, xy.y, zo.x);
    let opacity = sigmoid(zo.y);

    // Transform position to NDC space
    let pos_view = camera.view * vec4<f32>(pos_world, 1.0);
    let pos_clip = camera.proj * pos_view;
    let ndc_pos = pos_clip.xyz / pos_clip.w;

    // Frustum culling
    if (ndc_pos.x < -1.3|| ndc_pos.x > 1.3 ||
        ndc_pos.y < -1.3 || ndc_pos.y > 1.3 ||
        pos_view.z < 0.0) {
        return;
    }

    // Decode and normalize rotation quaternion
    let rotation = decodeRotation(gaussian.rot);

    // Decode scale vector
    let scale_vec4 = decodeScale(gaussian.scale);
    let scale = exp(scale_vec4.xyz);

    // Compute 3D covariance matrix
    let cov3D = computeCov3D(scale, rotation, render_settings.gaussian_scaling);

    // Compute 2D covariance matrix
    let cov2D = computeCov2D(pos_view, camera.focal.x, camera.focal.y, cov3D, camera.view);

    // Compute determinant and eigenvalues of the 2D covariance matrix
    let h_var = 0.3f;

    let a = cov2D.x + h_var;
    let b = cov2D.y;
    let c = cov2D.z + h_var;

    let det = a * c - b * b;
    if (det == 0.0) {
        return;
    }

    let det_inv = 1.0 / det;
    let conic = vec3<f32>(c * det_inv, -b * det_inv, a * det_inv);

    // Compute maximum radius
    let mid = 0.5 * (a + c);
    let lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
    let lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
    let radius = ceil(3.0 * sqrt(max(lambda1, lambda2)));

    // Compute camera position, direction and color
    let cam_pos = (camera.view_inv * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
    let dir = normalize(pos_world - cam_pos);
    let sh_deg = u32(render_settings.sh_deg);
    let color = computeColorFromSH(dir, idx, sh_deg);

    let splat_idx = atomicAdd(&sort_infos.keys_size, 1u);

    // Store the splat data
    splats[splat_idx].packedPos = pack2x16float(ndc_pos.xy);
    splats[splat_idx].packedSize = pack2x16float(vec2<f32>(radius, radius) / camera.viewport);
    splats[splat_idx].packedColor = array<u32,2>(pack2x16float(color.rg), pack2x16float(vec2<f32>(color.b, 1.0)));
    splats[splat_idx].packedConicAndOpacity = array<u32,2>(pack2x16float(conic.xy), pack2x16float(vec2<f32>(conic.z, opacity)));

    sort_depths[splat_idx] = bitcast<u32>(100 - pos_view.z);
    sort_indices[splat_idx] = splat_idx;

    let keys_per_dispatch = workgroupSize * sortKeyPerThread;
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
    if (splat_idx % keys_per_dispatch == 0u) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

fn decodeRotation(rot_data: array<u32,2>) -> vec4<f32> {
    let rot_xy = unpack2x16float(rot_data[0]);
    let rot_zw = unpack2x16float(rot_data[1]);
    return vec4<f32>(rot_xy.x, rot_xy.y, rot_zw.x, rot_zw.y);
}

fn decodeScale(scale_data: array<u32,2>) -> vec4<f32> {
    let scale_xy = unpack2x16float(scale_data[0]);
    let scale_zw = unpack2x16float(scale_data[1]);
    return vec4<f32>(scale_xy.x, scale_xy.y, scale_zw.x, scale_zw.y);
}

fn computeScalingMatrix(scale: vec3<f32>, scale_modifier: f32) -> mat3x3<f32> {
    let s = scale_modifier * scale;
    return mat3x3<f32>(
        vec3<f32>(s.x, 0.0, 0.0),
        vec3<f32>(0.0, s.y, 0.0),
        vec3<f32>(0.0, 0.0, s.z)
    );
}

fn computeRotationMatrix(q: vec4<f32>) -> mat3x3<f32> {
    let r = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;

    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let rx = r * x;
    let ry = r * y;
    let rz = r * z;

    // First column
    let c0 = vec3<f32>(
        1.0 - 2.0 * (yy + zz),
        2.0 * (xy - rz),
        2.0 * (xz + ry)
    );
    // Second column
    let c1 = vec3<f32>(
        2.0 * (xy + rz),
        1.0 - 2.0 * (xx + zz),
        2.0 * (yz - rx)
    );
    // Third column
    let c2 = vec3<f32>(
        2.0 * (xz - ry),
        2.0 * (yz + rx),
        1.0 - 2.0 * (xx + yy)
    );

    return mat3x3<f32>(c0, c1, c2);
}

fn computeCov3D(scale: vec3<f32>, q: vec4<f32>, gaussian_scaling: f32) -> array<f32,6> {
    // Compute scaling matrix S
    let S = computeScalingMatrix(scale, render_settings.gaussian_scaling);

    // Compute rotation matrix R
    let R = computeRotationMatrix(q);

    // Compute model matrix M
    let M = S * R;

    // Compute covariance matrix Sigma
    let Sigma = transpose(M) * M;

    // Compute 3D covariance matrix
    let cov3D = array<f32, 6>(
        Sigma[0][0],
        Sigma[0][1],
        Sigma[0][2],
        Sigma[1][1],
        Sigma[1][2],
        Sigma[2][2]
    );

    return cov3D;
}

fn computeCov2D(pos_view: vec4<f32>, focal_x: f32, focal_y: f32, cov3D: array<f32,6>, viewmatrix: mat4x4<f32>) -> vec3<f32> {
    let J = mat3x3f(
        focal_x / pos_view.z, 0.0, -focal_x * pos_view.x / (pos_view.z * pos_view.z),
        0.0, focal_y / pos_view.z, -focal_y * pos_view.y / (pos_view.z * pos_view.z),
        0.0, 0.0, 0.0
    );

    // Extract W matrix
    let W = transpose(mat3x3<f32>(
        viewmatrix[0].xyz,
        viewmatrix[1].xyz,
        viewmatrix[2].xyz
    ));

    // Compute T = W * J
    let T = W * J;

    // Build Vrk from cov3D
    let Vrk = mat3x3<f32>(
        vec3<f32>(cov3D[0], cov3D[1], cov3D[2]),
        vec3<f32>(cov3D[1], cov3D[3], cov3D[4]),
        vec3<f32>(cov3D[2], cov3D[4], cov3D[5])
    );

    // Compute the 2D covariance matrix: cov = transpose(T) * Vrk * T
    let cov = transpose(T) * Vrk * T;

    return vec3<f32>(cov[0][0], cov[0][1], cov[1][1]);
}