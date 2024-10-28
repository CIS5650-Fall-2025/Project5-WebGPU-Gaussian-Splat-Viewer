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

@group(0) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(0) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(0) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(0) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

@group(1) @binding(0)
var<uniform> camera: CameraUniforms;

@group(2) @binding(0)
var<storage, read> gaussians: array<Gaussian>;

@group(3) @binding(0)
var<storage, read_write> splats: array<Splat>;

@group(3) @binding(1)
var<uniform> render_settings: RenderSettings;

@group(3) @binding(2)
var<storage, read> colors: array<u32>;

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
// compute the base index
    let base_index = splat_idx * 24 + (c_idx / 2) * 3 + c_idx % 2;
    let color01 = unpack2x16float(colors[base_index + 0]);
    let color23 = unpack2x16float(colors[base_index + 1]);

    // unpack and return the color
    if (c_idx % 2 == 0) {
        return vec3f(color01.x, color01.y, color23.x);
    }

    return vec3f(color01.y, color23.x, color23.y);
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
    let posXY = unpack2x16float(gaussian.pos_opacity[0]);
    let posZAndOpacity = unpack2x16float(gaussian.pos_opacity[1]);
    let pos = vec3<f32>(posXY, posZAndOpacity.x);
    let opacity = 1.0f / (1.0f + exp(-posZAndOpacity.y)); // Sigmoid to bring back maximum opacity

    // Transform from world space to NDC space
    let viewPos = camera.view * vec4<f32>(pos, 1.0);
    var posNDC = camera.proj * viewPos;
    posNDC /= posNDC.w;

    // View-frustum culling of splats (treat frustum as 1.2x actual size to avoid culling edge splats)
    if (posNDC.x < -1.2 || posNDC.x > 1.2 || 
        posNDC.y < -1.2 || posNDC.y > 1.2 || 
        viewPos.z < 0.0) { // 100.0 is the far plane of the camera
        return;
    }

    // Unpack rotation (quaternion) and scale
    let rotWX = unpack2x16float(gaussian.rot[0]);
    let rotYZ = unpack2x16float(gaussian.rot[1]);
    // Scale is stored in log-space. Need to use exp to get the actual scale
    let scaleXY = exp(unpack2x16float(gaussian.scale[0]));
    let scaleZW = exp(unpack2x16float(gaussian.scale[1]));

    // Convert quaternion to rotation matrix
    let rotationMatrix = mat3x3f(
        1.0 - 2.0 * (rotYZ.x * rotYZ.x + rotYZ.y * rotYZ.y), 2.0 * (rotWX.y * rotYZ.x - rotWX.x * rotYZ.y), 2.0 * (rotWX.y * rotYZ.y + rotWX.x * rotYZ.x),
        2.0 * (rotWX.y * rotYZ.x + rotWX.x * rotYZ.y), 1.0 - 2.0 * (rotWX.y * rotWX.y + rotYZ.y * rotYZ.y), 2.0 * (rotYZ.x * rotYZ.y - rotWX.x * rotWX.y),
        2.0 * (rotWX.y * rotYZ.y - rotWX.x * rotYZ.x), 2.0 * (rotYZ.x * rotYZ.y + rotWX.x * rotWX.y), 1.0 - 2.0 * (rotWX.y * rotWX.y + rotYZ.x * rotYZ.x)
    );

    let scaleMatrix = mat3x3f(
        render_settings.gaussian_scaling * scaleXY.x, 0.0, 0.0,
        0.0, render_settings.gaussian_scaling * scaleXY.y, 0.0,
        0.0, 0.0, render_settings.gaussian_scaling * scaleZW.x
    );

    let covarianceMatrix3D = transpose(scaleMatrix * rotationMatrix) * scaleMatrix * rotationMatrix;

    // The following is derived from: https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu#L82-L97
    // Compute the Jacobian matrix for screen space transformation using viewPos
    let J = mat3x3f(
        camera.focal.x / viewPos.z, 0.0, -camera.focal.x * viewPos.x / (viewPos.z * viewPos.z),
        0.0, camera.focal.y / viewPos.z, -camera.focal.y * viewPos.y / (viewPos.z * viewPos.z),
        0.0, 0.0, 0.0
    );

    let W = transpose(mat3x3f(
        camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz
    ));

    let T = W * J;

    let V = mat3x3f(
        covarianceMatrix3D[0][0], covarianceMatrix3D[0][1], covarianceMatrix3D[0][2],
        covarianceMatrix3D[0][1], covarianceMatrix3D[1][1], covarianceMatrix3D[1][2],
        covarianceMatrix3D[0][2], covarianceMatrix3D[1][2], covarianceMatrix3D[2][2]
    );

    var covarianceMatrix2D = transpose(T) * transpose(V) * T;
    // Can think of the following as a low-pass filter: every Gaussian should be at least one pixel wide.
    covarianceMatrix2D[0][0] += 0.3f;
    covarianceMatrix2D[1][1] += 0.3f;

    let covariance2D = vec3f(
        covarianceMatrix2D[0][0], covarianceMatrix2D[0][1], covarianceMatrix2D[1][1]
    );

    var determinant = covariance2D.x * covariance2D.z - (covariance2D.y * covariance2D.y);
    if (determinant == 0.0f) {
        return;
    }

    // Compute radius
    let mid = (covariance2D.x + covariance2D.z) * 0.5;
    let lambda1 = mid + sqrt(max(0.1, mid * mid - determinant));
    let lambda2 = mid - sqrt(max(0.1, mid * mid - determinant));
    let radius = ceil(3.0f * sqrt(max(lambda1, lambda2)));

    // We also need the viewing direction (camera's position to the splat's position) so we can get its color
    // We have the camera's position from the view matrix.
    let camPos = -camera.view[3].xyz;
    let direction = normalize(pos - camPos);
    let color = computeColorFromSH(direction, idx, u32(render_settings.sh_deg));

    // Finally, compute the conic (used to determine if a fragment is inside the Gaussian)
    let det_inv = 1.0 / determinant;
    let conic = vec3f(
        covariance2D.z * det_inv,
        -covariance2D.y * det_inv,
        covariance2D.x * det_inv
    );

    // Now that we have the splat data, we need the index to store it
    let sortKeysIndex = atomicAdd(&sort_infos.keys_size, 1);
    splats[sortKeysIndex].packedPos = pack2x16float(posNDC.xy);
    splats[sortKeysIndex].packedSize = pack2x16float(vec2f(radius, radius) / camera.viewport);
    splats[sortKeysIndex].packedColor[0] = pack2x16float(color.rg);
    splats[sortKeysIndex].packedColor[1] = pack2x16float(vec2f(color.b, 1.0f));
    splats[sortKeysIndex].packedConicAndOpacity[0] = pack2x16float(conic.xy); 
    splats[sortKeysIndex].packedConicAndOpacity[1] = pack2x16float(vec2f(conic.z, opacity));

    // 100.0 is the far plane of the camera (hardcoded in renderer.ts)
    sort_depths[sortKeysIndex] = bitcast<u32>(100.0 - viewPos.z); // Bitcast because the radix sort operates on u32s
    sort_indices[sortKeysIndex] = sortKeysIndex;

    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 

    // Increment the number of work groups we dispatch if we've filled up the current one
    if (sortKeysIndex % keys_per_dispatch == 0) {
        atomicAdd(&sort_dispatch.dispatch_x, 1);
    }
}