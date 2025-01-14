enable f16;

const SH_C0: f16 = 0.28209479177387814;
const SH_C1: f16 = 0.4886025119029199;
const SH_C2 = array<f16, 5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f16, 7>(
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
    dispatchX: atomic<u32>,
    dispatchY: u32,
    dispatchZ: u32,
}

struct SortInfos {
    keysSize: atomic<u32>,  // instance_count in DrawIndirect
    //data below is for info inside radix sort 
    paddedSize: u32, 
    passes: u32,
    evenPass: u32,
    oddPass: u32,
}

struct CameraUniforms {
    view: mat4x4f,
    viewInv: mat4x4f,
    proj: mat4x4f,
    projInv: mat4x4f,
    viewport: vec2f,
    focal: vec2f,
    clippingPlanes: vec2f
};

struct RenderSettings {
    gaussianScaling: f32,
    shDegree: f32,
}

struct Gaussian {
    pos_opacity: vec2u,
    rot: vec2u,
    scale: vec2u
};

struct Splat {
    pos: u32,
    size: u32,
    color_opacity: vec2u,
    conic: vec2u
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<uniform> renderSettings: RenderSettings;

@group(1) @binding(0)
var<storage, read> gaussians: array<Gaussian>;
@group(1) @binding(1)
var<storage, read> shCoefs: array<u32>;
@group(1) @binding(2)
var<storage, read_write> splats: array<Splat>;

@group(2) @binding(0)
var<storage, read_write> sortInfos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sortDepths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sortIndices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sortDispatch: DispatchIndirect;

// reads the ith sh coef from the storage buffer 
fn readSHCoef(splatIdx: u32, coefIdx: u32) -> vec3h {
    const maxNumCoefs = 16u;
    let offset = (splatIdx * maxNumCoefs + coefIdx) * 3;
    if (offset % 2 == 0) {
        return vec3h(
            bitcast<vec2h>(shCoefs[offset / 2]),
            bitcast<vec2h>(shCoefs[offset / 2 + 1]).x
        );
    }
    else {
        return vec3h(
            bitcast<vec2h>(shCoefs[offset / 2]).y,
            bitcast<vec2h>(shCoefs[offset / 2 + 1])
        );
    }
}

fn sigmoid(x: f16) -> f16 {
    return 1.0 / (1.0 + exp(-x));
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3h, v_idx: u32, sh_deg: u32) -> vec3h {
    var result = SH_C0 * readSHCoef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * readSHCoef(v_idx, 1u)
                  + SH_C1 * z * readSHCoef(v_idx, 2u)
                  - SH_C1 * x * readSHCoef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * readSHCoef(v_idx, 4u)
                    + SH_C2[1] * yz * readSHCoef(v_idx, 5u)
                    + SH_C2[2] * (2.0 * zz - xx - yy) * readSHCoef(v_idx, 6u)
                    + SH_C2[3] * xz * readSHCoef(v_idx, 7u)
                    + SH_C2[4] * (xx - yy) * readSHCoef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * readSHCoef(v_idx, 9u)
                        + SH_C3[1] * xy * z * readSHCoef(v_idx, 10u)
                        + SH_C3[2] * y * (4.0 * zz - xx - yy) * readSHCoef(v_idx, 11u)
                        + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * readSHCoef(v_idx, 12u)
                        + SH_C3[4] * x * (4.0 * zz - xx - yy) * readSHCoef(v_idx, 13u)
                        + SH_C3[5] * z * (xx - yy) * readSHCoef(v_idx, 14u)
                        + SH_C3[6] * x * (xx - 3.0 * yy) * readSHCoef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return  max(vec3h(0.0), result);
}

@compute @workgroup_size(workgroupSize, 1, 1)
fn preprocess(@builtin(global_invocation_id) globalIndex: vec3u) {
    let index = globalIndex.x;
    if (index >= arrayLength(&gaussians)) { return; }

    let gaussian = gaussians[index];

    // Get position in screen space
    let pos_opac = vec4f(unpack2x16float(gaussian.pos_opacity[0]), unpack2x16float(gaussian.pos_opacity[1]));
    let posView = camera.view * vec4f(pos_opac.xyz, 1.0);
    // Z-clipping
    if (posView.z < camera.clippingPlanes[0] || posView.z > camera.clippingPlanes[1]) { return; }
    let posClip = mat4x4h(camera.proj) * vec4h(posView);
    let posNDC = posClip.xyz / posClip.w;
    let depth = posView.z;

    // Simple view frustum culling
    if (any(abs(posNDC.xy) > vec2h(1.2))) { return; }

    // Set some f16 variables to prevent repeated casting
    let focal = vec2h(camera.focal);
    let viewport = vec2h(camera.viewport);
    let posView16 = vec3h(posView.xyz);

    // Compute color from spherical harmonics
    // Camera world position is last column of inversed view matrix
    let direction = normalize(vec3h(pos_opac.xyz - camera.viewInv[3].xyz));
    let color = computeColorFromSH(
        direction, index, u32(renderSettings.shDegree)
    );
    let opacity = sigmoid(f16(pos_opac.w));

    // Calculate 3D covariance matrix from scale and rotation
    // rotation matrix
    let rot = vec4h(bitcast<vec2h>(gaussian.rot[0]), bitcast<vec2h>(gaussian.rot[1]));
    let rotMat = mat3x3h(
        1.0 - 2.0 * (rot.z * rot.z + rot.w * rot.w), 2.0 * (rot.y * rot.z + rot.x * rot.w), 2.0 * (rot.y * rot.w - rot.x * rot.z),
        2.0 * (rot.y * rot.z - rot.x * rot.w), 1.0 - 2.0 * (rot.y * rot.y + rot.w * rot.w), 2.0 * (rot.z * rot.w + rot.x * rot.y),
        2.0 * (rot.y * rot.w + rot.x * rot.z), 2.0 * (rot.z * rot.w - rot.x * rot.y), 1.0 - 2.0 * (rot.y * rot.y + rot.z * rot.z)
    );
    // diagonal scale matrix, saved in log space
    let scale = f16(renderSettings.gaussianScaling) * exp(vec3h(bitcast<vec2h>(gaussian.scale[0]), bitcast<vec2h>(gaussian.scale[1]).x));
    var tmpMat = mat3x3h(
        scale.x * rotMat[0],
        scale.y * rotMat[1],
        scale.z * rotMat[2],
    );
    let cov3d = tmpMat * transpose(tmpMat);

    // Calculate 2D covariance matrix from 3D covariance matrix
    let W = mat3x3h(mat3x3f(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz));
    
    let lim = 0.65 * viewport * focal;
    let t = vec3h(
        clamp(posView16.xy / posView16.z, -lim, lim) * posView16.z,
        posView16.z
    );
    let invTz = 1.0 / t.z;
    let J = mat3x2h(
		               focal.x * invTz,                            0.0,
                                   0.0,                focal.y * invTz, 
        -focal.x * t.x * invTz * invTz, -focal.y * t.y * invTz * invTz
    );

    let tmpMat2 = J * W;
    var cov2d = tmpMat2 * cov3d * transpose(tmpMat2);
    cov2d[0][0] += 0.3;
    cov2d[1][1] += 0.3;

    // Calculate max radius via eigenvalues of 2D covariance matrix
    // Use f32 for some intermediate calculations
    let cov2d32 = mat2x2f(cov2d);
    let det = determinant(cov2d32);

	let mid = 0.5 * (cov2d32[0][0] + cov2d32[1][1]);
    let dist = sqrt(max(0.1, mid * mid - det));
	let lambda1 = f16(mid + dist);
	let lambda2 = f16(mid - dist);

    // Get size from max radius in NDC space
	let size = ceil(3.0 * sqrt(max(lambda1, lambda2))) * 2.0 / viewport;

    // Store sorting information
    let keyIndex = atomicAdd(&sortInfos.keysSize, 1);
    sortDepths[keyIndex] = bitcast<u32>(camera.clippingPlanes[1] - depth);
    sortIndices[keyIndex] = keyIndex;

    // increment sortDispatch.dispatchX each time you reach limit for one dispatch of keys
    if (keyIndex % (workgroupSize * sortKeyPerThread) == 0) {
        atomicAdd(&sortDispatch.dispatchX, 1);
    }
    
    // Handle output to render pipeline
    var output: Splat;
    output.pos = bitcast<u32>(posNDC.xy);
    output.size = bitcast<u32>(size);
    output.color_opacity = vec2u(
        bitcast<u32>(color.rg),
        bitcast<u32>(vec2h(color.b, opacity))
    );
    let detInv = 1.0 / det;
    output.conic = vec2u(
        pack2x16float(vec2f(detInv * cov2d32[1][1], detInv * -cov2d32[0][1])),
        pack2x16float(vec2f(detInv * cov2d32[0][0], 0))
    );
    splats[keyIndex] = output;
}