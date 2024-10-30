// Spherical Harmonic constants used in shading computations
const SH_C0: f32 = 0.28209479177387814;
const SH_C1: f32 = 0.4886025119029199;
const SH_C2: array<f32, 5> = array<f32, 5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3: array<f32, 7> = array<f32, 7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

// Overridable constants for workgroup configuration
override workgroupSize: u32;          // Number of threads per workgroup
override sortKeyPerThread: u32;       // Number of sort keys processed per thread

// Structures for dispatching compute shader workgroups and sorting information
struct DispatchIndirect {
    dispatch_x: atomic<u32>,  // Atomic counter for workgroups in the X dimension
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,   // Total number of keys to sort (instance count)
    padded_size: u32,         // Padded size of the keys array for radix sort
    passes: u32,              // Number of passes in radix sort
    even_pass: u32,
    odd_pass: u32,
}

// Uniforms for camera transformations and parameters
struct CameraUniforms {
    view: mat4x4<f32>,        // View matrix
    view_inv: mat4x4<f32>,    // Inverse view matrix
    proj: mat4x4<f32>,        // Projection matrix
    proj_inv: mat4x4<f32>,    // Inverse projection matrix
    viewport: vec2<f32>,      // Viewport dimensions
    focal: vec2<f32>          // Focal lengths in x and y directions
}

// Uniforms for rendering settings
struct RenderSettings {
    gaussian_scaling: f32,    // Scaling factor for Gaussian splats
    sh_deg: f32,              // Degree of spherical harmonics
}

// Structures representing input Gaussians and output splats
struct Gaussian {
    pos_opa: array<u32, 2>,  // Packed position and opacity
    rot: array<u32, 2>,          // Packed rotation quaternion
    scale: array<u32, 2>         // Packed scale factors in log-space
}

struct Splat {
    pos: u32,                  // Packed NDC position
    size: u32,                 // Packed size (radius)
    color: array<u32, 2>,      // Packed RGB color
    conic_opa: array<u32, 2>  // Packed conic coefficients and opacity
}

// Shader resource bindings
@group(0) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(0) @binding(1)
var<storage, read_write> sort_depths: array<u32>;
@group(0) @binding(2)
var<storage, read_write> sort_indices: array<u32>;
@group(0) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;
@group(0) @binding(4)
var<uniform> camera: CameraUniforms;
@group(0) @binding(5)
var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(6)
var<storage, read_write> splats: array<Splat>;
@group(0) @binding(7)
var<uniform> render_settings: RenderSettings;
@group(0) @binding(8)
var<storage, read> colors: array<u32>;

/// Reads the spherical harmonic coefficient for a given splat index and coefficient index
fn readSHCoefficient(splat_idx: u32, coeff_idx: u32) -> vec3<f32> {
    // Calculate the base index in the color array
    let base_index = splat_idx * 24 + (coeff_idx / 2) * 3 + coeff_idx % 2;
    let color_part1 = unpack2x16float(colors[base_index + 0]);
    let color_part2 = unpack2x16float(colors[base_index + 1]);

    // Return the appropriate color vector based on the coefficient index
    if (coeff_idx % 2 == 0) {
        return vec3f(color_part1.x, color_part1.y, color_part2.x);
    }

    return vec3f(color_part1.y, color_part2.x, color_part2.y);
}

// Computes the color from spherical harmonics using the Condon–Shortley phase
fn computeColorFromSH(direction: vec3<f32>, splat_idx: u32, sh_degree: u32) -> vec3<f32> {
    var result = SH_C0 * readSHCoefficient(splat_idx, 0u);

    if sh_degree > 0u {
        let x = direction.x;
        let y = direction.y;
        let z = direction.z;

        result += -SH_C1 * y * readSHCoefficient(splat_idx, 1u)
                  + SH_C1 * z * readSHCoefficient(splat_idx, 2u)
                  - SH_C1 * x * readSHCoefficient(splat_idx, 3u);

        if sh_degree > 1u {
            let xx = x * x;
            let yy = y * y;
            let zz = z * z;
            let xy = x * y;
            let yz = y * z;
            let xz = x * z;

            result += SH_C2[0] * xy * readSHCoefficient(splat_idx, 4u)
                      + SH_C2[1] * yz * readSHCoefficient(splat_idx, 5u)
                      + SH_C2[2] * (2.0 * zz - xx - yy) * readSHCoefficient(splat_idx, 6u)
                      + SH_C2[3] * xz * readSHCoefficient(splat_idx, 7u)
                      + SH_C2[4] * (xx - yy) * readSHCoefficient(splat_idx, 8u);

            if sh_degree > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * readSHCoefficient(splat_idx, 9u)
                          + SH_C3[1] * xy * z * readSHCoefficient(splat_idx, 10u)
                          + SH_C3[2] * y * (4.0 * zz - xx - yy) * readSHCoefficient(splat_idx, 11u)
                          + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * readSHCoefficient(splat_idx, 12u)
                          + SH_C3[4] * x * (4.0 * zz - xx - yy) * readSHCoefficient(splat_idx, 13u)
                          + SH_C3[5] * z * (xx - yy) * readSHCoefficient(splat_idx, 14u)
                          + SH_C3[6] * x * (xx - 3.0 * yy) * readSHCoefficient(splat_idx, 15u);
            }
        }
    }
    result += 0.5; // Offset to adjust the color range

    return max(vec3<f32>(0.0), result);
}

@compute @workgroup_size(workgroupSize, 1, 1)
fn preprocess(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let splat_idx = global_id.x;
    if (splat_idx >= arrayLength(&gaussians)) {
        return;
    }

    // Retrieve Gaussian data
    let gaussian = gaussians[splat_idx];
    let pos_xy = unpack2x16float(gaussian.pos_opa[0]);
    let pos_z_opacity = unpack2x16float(gaussian.pos_opa[1]);
    let position = vec3<f32>(pos_xy, pos_z_opacity.x);

    // Decode opacity using a sigmoid function
    let opacity = 1.0 / (1.0 + exp(-pos_z_opacity.y));

    // Transform position from world space to Normalized Device Coordinates (NDC)
    let view_pos = camera.view * vec4<f32>(position, 1.0);
    var pos_ndc = camera.proj * view_pos;
    pos_ndc /= pos_ndc.w;

    // Frustum culling with a slight margin to avoid clipping edge splats
    if (pos_ndc.x < -1.2 || pos_ndc.x > 1.2 ||
        pos_ndc.y < -1.2 || pos_ndc.y > 1.2 ||
        view_pos.z < 0.0) {
        return;
    }

    // Unpack rotation quaternion components
    let rot_wx = unpack2x16float(gaussian.rot[0]);
    let rot_yz = unpack2x16float(gaussian.rot[1]);

    // Unpack and exponentiate scale factors (stored in log-space)
    let scale_xy = exp(unpack2x16float(gaussian.scale[0]));
    let scale_zw = exp(unpack2x16float(gaussian.scale[1]));

    // Construct rotation matrix from quaternion
    let rotation_matrix = mat3x3<f32>(
        1.0 - 2.0 * (rot_yz.x * rot_yz.x + rot_yz.y * rot_yz.y),
        2.0 * (rot_wx.y * rot_yz.x - rot_wx.x * rot_yz.y),
        2.0 * (rot_wx.y * rot_yz.y + rot_wx.x * rot_yz.x),
        2.0 * (rot_wx.y * rot_yz.x + rot_wx.x * rot_yz.y),
        1.0 - 2.0 * (rot_wx.y * rot_wx.y + rot_yz.y * rot_yz.y),
        2.0 * (rot_yz.x * rot_yz.y - rot_wx.x * rot_wx.y),
        2.0 * (rot_wx.y * rot_yz.y - rot_wx.x * rot_yz.x),
        2.0 * (rot_yz.x * rot_yz.y + rot_wx.x * rot_wx.y),
        1.0 - 2.0 * (rot_wx.y * rot_wx.y + rot_yz.x * rot_yz.x)
    );

    // Create scale matrix
    let scale_matrix = mat3x3<f32>(
        render_settings.gaussian_scaling * scale_xy.x, 0.0, 0.0,
        0.0, render_settings.gaussian_scaling * scale_xy.y, 0.0,
        0.0, 0.0, render_settings.gaussian_scaling * scale_zw.x
    );

    // Compute 3D covariance matrix for the Gaussian
    let covariance_matrix_3d = transpose(scale_matrix * rotation_matrix) * scale_matrix * rotation_matrix;

    // Compute the Jacobian matrix for screen space transformation
    let jacobian_j = mat3x3<f32>(
        camera.focal.x / view_pos.z, 0.0, -camera.focal.x * view_pos.x / (view_pos.z * view_pos.z),
        0.0, camera.focal.y / view_pos.z, -camera.focal.y * view_pos.y / (view_pos.z * view_pos.z),
        0.0, 0.0, 0.0
    );

    // Extract rotation part of the camera view matrix
    let camera_rotation = transpose(mat3x3<f32>(
        camera.view[0].xyz,
        camera.view[1].xyz,
        camera.view[2].xyz
    ));

    // Compute transformation matrix
    let transformation_t = camera_rotation * jacobian_j;

    // Construct symmetric covariance matrix
    let covariance_v = mat3x3<f32>(
        covariance_matrix_3d[0][0], covariance_matrix_3d[0][1], covariance_matrix_3d[0][2],
        covariance_matrix_3d[0][1], covariance_matrix_3d[1][1], covariance_matrix_3d[1][2],
        covariance_matrix_3d[0][2], covariance_matrix_3d[1][2], covariance_matrix_3d[2][2]
    );

    // Compute 2D covariance matrix in screen space
    var covariance_matrix_2d = transpose(transformation_t) * transpose(covariance_v) * transformation_t;

    // Apply minimum variance to prevent undersized Gaussians
    covariance_matrix_2d[0][0] += 0.3;
    covariance_matrix_2d[1][1] += 0.3;

    // Extract unique elements from the symmetric covariance matrix
    let covariance_2d = vec3<f32>(
        covariance_matrix_2d[0][0],
        covariance_matrix_2d[0][1],
        covariance_matrix_2d[1][1]
    );

    // Calculate the determinant of the covariance matrix
    var determinant = covariance_2d.x * covariance_2d.z - (covariance_2d.y * covariance_2d.y);
    if (determinant == 0.0) {
        return;
    }

    // Compute eigenvalues to determine the radius
    let mid = (covariance_2d.x + covariance_2d.z) * 0.5;
    let sqrt_term = sqrt(max(0.1, mid * mid - determinant));
    let lambda1 = mid + sqrt_term;
    let lambda2 = mid - sqrt_term;
    let radius = ceil(3.0 * sqrt(max(lambda1, lambda2)));

    // Calculate the viewing direction for shading
    let camera_position = -camera.view[3].xyz;
    let view_direction = normalize(position - camera_position);
    let color = computeColorFromSH(view_direction, splat_idx, u32(render_settings.sh_deg));

    // Compute conic coefficients for fragment coverage determination
    let determinant_inv = 1.0 / determinant;
    let conic_coefficients = vec3<f32>(
        covariance_2d.z * determinant_inv,
        -covariance_2d.y * determinant_inv,
        covariance_2d.x * determinant_inv
    );

    // Get index to store the computed splat
    let splat_store_index = atomicAdd(&sort_infos.keys_size, 1);

    // Store splat data for rendering
    splats[splat_store_index].pos = pack2x16float(pos_ndc.xy);
    splats[splat_store_index].size = pack2x16float(vec2<f32>(radius, radius) / camera.viewport);
    splats[splat_store_index].color[0] = pack2x16float(color.rg);
    splats[splat_store_index].color[1] = pack2x16float(vec2<f32>(color.b, 1.0));
    splats[splat_store_index].conic_opa[0] = pack2x16float(conic_coefficients.xy);
    splats[splat_store_index].conic_opa[1] = pack2x16float(vec2<f32>(conic_coefficients.z, opacity));

    // Store depth for sorting (bitcast as u32 for radix sort)
    sort_depths[splat_store_index] = bitcast<u32>(100.0 - view_pos.z);
    sort_indices[splat_store_index] = splat_store_index;

    let keys_per_dispatch = workgroupSize * sortKeyPerThread;

    // Increment dispatch count if necessary
    if (splat_store_index % keys_per_dispatch == 0) {
        atomicAdd(&sort_dispatch.dispatch_x, 1);
    }
}
