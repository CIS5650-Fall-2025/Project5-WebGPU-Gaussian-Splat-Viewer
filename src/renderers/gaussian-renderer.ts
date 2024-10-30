import {PointCloud} from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import {get_sorter, c_histogram_block_rows, C} from '../sort/sort';
import {Renderer} from './renderer';

export interface GaussianRenderer extends Renderer {
    render_settings_buffer: GPUBuffer,
}

// Utility to create GPU buffers
const createBuffer = (
    device: GPUDevice,
    label: string,
    size: number,
    usage: GPUBufferUsageFlags,
    data?: ArrayBuffer | ArrayBufferView
) => {
    const buffer = device.createBuffer({label, size, usage});
    if (data) device.queue.writeBuffer(buffer, 0, data);
    return buffer;
};

export default function get_renderer(
    pc: PointCloud,
    device: GPUDevice,
    presentation_format: GPUTextureFormat,
    camera_buffer: GPUBuffer,
): GaussianRenderer {

    const sorter = get_sorter(pc.num_points, device);

    // ===============================================
    //            Initialize GPU Buffers
    // ===============================================

    const nulling_data = new Uint32Array([0]);
    const nulling_buffer = createBuffer(
        device,
        'nulling buffer',
        4,
        GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        nulling_data,
    );

    const splatSize = 24;
    const splat_buffer = createBuffer(
        device,
        'splat buffer',
        splatSize * pc.num_points,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    );

    const render_settings_buffer = createBuffer(
        device,
        'render settings buffer',
        8,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        new Float32Array([1.0, pc.sh_deg]),
    );

    const indirect_draw_buffer = createBuffer(
        device,
        'indirect draw buffer',
        4 * 4,
        GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
        new Uint32Array([6, 0, 0, 0]),
    );

    // ===============================================
    //    Create Compute Pipeline and Bind Groups
    // ===============================================

    const preprocess_pipeline = device.createComputePipeline({
        label: 'preprocess',
        layout: 'auto',
        compute: {
            module: device.createShaderModule({code: preprocessWGSL}),
            entryPoint: 'preprocess',
            constants: {
                workgroupSize: C.histogram_wg_size,
                sortKeyPerThread: c_histogram_block_rows,
            },
        },
    });

    const sort_bind_group = device.createBindGroup({
        label: 'sort',
        layout: preprocess_pipeline.getBindGroupLayout(2),
        entries: [
            {binding: 0, resource: {buffer: sorter.sort_info_buffer}},
            {binding: 1, resource: {buffer: sorter.ping_pong[0].sort_depths_buffer}},
            {binding: 2, resource: {buffer: sorter.ping_pong[0].sort_indices_buffer}},
            {binding: 3, resource: {buffer: sorter.sort_dispatch_indirect_buffer}},
        ],
    });

    const camera_bind_group = device.createBindGroup({
        label: 'camera',
        layout: preprocess_pipeline.getBindGroupLayout(0),
        entries: [
            {binding: 0, resource: {buffer: camera_buffer}},
            {binding: 1, resource: {buffer: render_settings_buffer}},
        ],
    });

    const guassian_bind_group = device.createBindGroup({
        label: 'gaussian',
        layout: preprocess_pipeline.getBindGroupLayout(1),
        entries: [
            {binding: 0, resource: {buffer: pc.gaussian_3d_buffer}},
            {binding: 1, resource: {buffer: pc.sh_buffer}},
        ],
    });

    const splat_bind_group = device.createBindGroup({
        label: 'splat',
        layout: preprocess_pipeline.getBindGroupLayout(3),
        entries: [
            {binding: 0, resource: {buffer: splat_buffer}},
        ],
    });

    // ===============================================
    //    Create Render Pipeline and Bind Groups
    // ===============================================

    const render_pipeline = device.createRenderPipeline({
        label: 'Render Pipeline',
        layout: 'auto',
        vertex: {
            module: device.createShaderModule({code: renderWGSL}),
            entryPoint: 'vs_main',
        },
        fragment: {
            module: device.createShaderModule({code: renderWGSL}),
            entryPoint: 'fs_main',
            targets: [{
                format: presentation_format,
                blend: {
                    color: {
                        srcFactor: 'one',
                        dstFactor: 'one-minus-src-alpha',
                        operation: 'add',
                    },
                    alpha: {
                        srcFactor: 'one',
                        dstFactor: 'one-minus-src-alpha',
                        operation: 'add',
                    },
                },
            }],
        }
    });

    const render_bind_group = device.createBindGroup({
        label: 'render',
        layout: render_pipeline.getBindGroupLayout(0),
        entries: [
            {binding: 0, resource: {buffer: camera_buffer}},
            {binding: 1, resource: {buffer: splat_buffer}},
            {binding: 2, resource: {buffer: sorter.ping_pong[0].sort_indices_buffer}},
        ],
    });

    // ===============================================
    //    Command Encoder Functions
    // ===============================================

    const computePass = (encoder: GPUCommandEncoder) => {
        // Create a compute pass
        const computePass = encoder.beginComputePass();
        computePass.setPipeline(preprocess_pipeline);

        // Set bind groups
        computePass.setBindGroup(0, camera_bind_group);
        computePass.setBindGroup(1, guassian_bind_group);
        computePass.setBindGroup(2, sort_bind_group);
        computePass.setBindGroup(3, splat_bind_group);

        // Dispatch compute shader
        const numWorkgroups = Math.ceil(pc.num_points / C.histogram_wg_size);
        computePass.dispatchWorkgroups(numWorkgroups);

        computePass.end();
    }

    const renderPass = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
        // Render Pass
        const renderPass = encoder.beginRenderPass({
            colorAttachments: [{
                view: texture_view,
                clearValue: {r: 0, g: 0, b: 0, a: 1},
                loadOp: 'clear',
                storeOp: 'store',
            }]
        });

        renderPass.setPipeline(render_pipeline);
        renderPass.setBindGroup(0, render_bind_group);
        renderPass.drawIndirect(indirect_draw_buffer, 0);

        renderPass.end();
    }

    // ===============================================
    //    Return Render Object
    // ===============================================
    return {
        frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
            // Reset buffers at the start of each frame
            encoder.copyBufferToBuffer(nulling_buffer, 0, sorter.sort_info_buffer, 0, 4);
            encoder.copyBufferToBuffer(nulling_buffer, 0, sorter.sort_dispatch_indirect_buffer, 0, 4);

            // Compute Pass
            computePass(encoder);

            // Proceed with sorting and rendering
            sorter.sort(encoder);

            // Set indirect draw buffer
            encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, indirect_draw_buffer, 4, 4);

            // Render Pass
            renderPass(encoder, texture_view);
        },
        camera_buffer,
        render_settings_buffer,
    };
}
