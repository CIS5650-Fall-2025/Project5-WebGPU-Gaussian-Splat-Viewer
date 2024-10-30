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
    position: vec2<f32>, 
    size: vec2<f32>,          
    color: vec3<f32>, 
};

//TODO: bind your data here
@group(0) @binding(0)
var<storage, read_write> splatBuffer: array<Splat>;

@group(0) @binding(1)
var<storage, read> sh_buffer: array<vec3<f32>>;

@group(0) @binding(2)
var<storage, read> gaussian3d_buffer: array<Gaussian>;


@group(1) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(1)
var<uniform> renderSettings: RenderSettings;



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
   return vec3<f32>(1.0);
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

    if(idx >= arrayLength(&gaussian3d_buffer)){
        return;
    }

    let gaussian = gaussian3d_buffer[idx];
    let xy = unpack2x16float(gaussian.pos_opacity[0]);
    let z = unpack2x16float(gaussian.pos_opacity[1]);
    let xyz = vec3<f32>(xy, z.x);
    let alpha = z.y;

    var positionNDC = camera.proj * (camera.view * vec4<f32>(xyz, 1.0));
    
    positionNDC/=positionNDC.w;
    let boundary  = 1.2f;

    if(positionNDC.x < -boundary || positionNDC.x> boundary ||positionNDC.y < -boundary || positionNDC.y > boundary 
    || positionNDC.z < 0.0 || positionNDC.z > 1.0){
        return;
    }

    //calculatin of  3D covariance
    let quatWX = unpack2x16float(gaussian.rot[0]);
    let quatYZ = unpack2x16float(gaussian.rot[1]);
    let scaleXY = unpack2x16float(gaussian.scale[0]);
    let scaleZ = unpack2x16float(gaussian.scale[1]);
    let scale = exp(vec3<f32>(
        scaleXY.x, 
        scaleXY.y, 
        scaleZ.x));


    let rotationMatrix = mat3x3<f32>(
        1.0f - 2.0f * (quatYZ.x * quatYZ.x + quatYZ.y * quatYZ.y), 2.0f * (quatWX.y * quatYZ.x - quatWX.x * quatYZ.y), 2.0f * (quatWX.y * quatYZ.y + quatWX.x * quatYZ.x),
        2.0f * (quatWX.y * quatYZ.x + quatWX.x * quatYZ.y), 1.0f - 2.0f * (quatWX.y * quatWX.y + quatYZ.y * quatYZ.y), 2.0f * (quatYZ.x * quatYZ.y - quatWX.x * quatWX.y),
        2.0f * (quatWX.y * quatYZ.y - quatWX.x * quatYZ.x), 2.0f * (quatYZ.x * quatYZ.y + quatWX.x * quatWX.y), 1.0f - 2.0f * (quatWX.y * quatWX.y + quatYZ.x * quatYZ.x)
    );

    let scaleMatrix = mat3x3<f32>(
        scale.x * renderSettings.gaussian_scaling , 0.0, 0.0,
        0.0, scale.y * renderSettings.gaussian_scaling , 0.0,
        0.0, 0.0, scale.z * renderSettings.gaussian_scaling ,
    );

    let covariance3D  = transpose(scaleMatrix * rotationMatrix) * scaleMatrix * rotationMatrix;
    let covarianceSymmetric = array<f32, 6>(covariance3D[0][0], covariance3D[0][1], covariance3D[0][2],
                                            covariance3D[1][1], covariance3D[1][2], covariance3D[2][2],);

    //2D covariance matrix
    var t = (camera.view * vec4<f32>(xyz, 1.0)).xyz;


    let J = mat3x3<f32>(
        camera.focal.x / t.z, 0.0f, -(camera.focal.x * t.x)/(t.z * t.z),
        0.0f, camera.focal.y / t.z, -(camera.focal.y * t.y)/(t.x * t.z),
        0.0, 0.0, 0.0
    );

    let W = mat3x3<f32>(
        camera.view[0].x,  camera.view[1].x, camera.view[2].x,
        camera.view[0].y,  camera.view[1].y, camera.view[2].y,
        camera.view[0].z,  camera.view[1].z, camera.view[2].z
    );

    let T = W * J;

    let Vrk = mat3x3<f32>(
        covarianceSymmetric[0], covarianceSymmetric[1], covarianceSymmetric[2],
        covarianceSymmetric[1], covarianceSymmetric[3], covarianceSymmetric[4],
        covarianceSymmetric[2], covarianceSymmetric[4], covarianceSymmetric[5],
    );

    var cov = transpose(T) * transpose(Vrk) * T;
    cov[0][0] += 0.3f;
    cov[1][1] += 0.3f;

    let cov_2D = vec3<f32>(cov[0][0], cov[0][1], cov[1][1]);
    //radius
    let det = (cov_2D.x * cov_2D.z - cov_2D.y * cov_2D.y);

    let mid = 0.5f * (cov_2D.x + cov_2D.z);
    let lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
    let lambda2 = mid - sqrt(max(0.1f, mid * mid - det));

    let radius = ceil(3.0f * sqrt(max(lambda1, lambda2)));

    let maxNDCsize = vec2( 2.0f * radius, 2.0f * radius)/camera.viewport;


    //testing
    
    let testing2 = sh_buffer[0];
    let render = renderSettings.gaussian_scaling;
    let testing3 = sort_depths[0];
    let testing4 = sort_indices[0];
    let testing5 = sort_dispatch.dispatch_y;

    let index = atomicAdd(&sort_infos.keys_size, 1u);
    sort_indices[index] = index;
    splatBuffer[index].position = positionNDC.xy;
    splatBuffer[index].size = maxNDCsize;

    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 
    if (index % keys_per_dispatch == 0){
        atomicAdd(&sort_dispatch.dispatch_x, 1);
    }
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
}
