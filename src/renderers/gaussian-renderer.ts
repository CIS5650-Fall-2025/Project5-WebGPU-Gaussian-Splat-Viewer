import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter, c_histogram_block_rows, C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {

}

// Utility to create GPU buffers
const createBuffer = (
    device: GPUDevice,
    label: string,
    size: number,
    usage: GPUBufferUsageFlags,
    data?: ArrayBuffer | ArrayBufferView
) => {
    const buffer = device.createBuffer({ label, size, usage });
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

    // ===============================================
    //    Create Compute Pipeline and Bind Groups
    // ===============================================
    const preprocess_pipeline = device.createComputePipeline({
        label: 'preprocess',
        layout: 'auto',
        compute: {
            module: device.createShaderModule({ code: preprocessWGSL }),
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
            { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
            { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
            { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
            { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
        ],
    });


    // ===============================================
    //    Create Render Pipeline and Bind Groups
    // ===============================================
    const render_shader = device.createShaderModule({ code: renderWGSL });
    const render_pipeline = device.createRenderPipeline({
        label: 'gaussian render pipeline',
        layout: 'auto',
        vertex: {
            module: render_shader,
            entryPoint: 'vs_main',
            // buffers: [
            //     {
            //         arrayStride: 2 * 4,
            //         attributes: [
            //             { shaderLocation: 0, offset: 0, format: 'float32x2' },
            //         ],
            //     },
            // ],
        },
        fragment: {
            module: render_shader,
            entryPoint: 'fs_main',
            targets: [{ format: presentation_format }],
        },
        primitive: {
            topology: "triangle-strip"
        }
    });

    const camera_bind_group = device.createBindGroup({
        label: 'gaussian camera',
        layout: render_pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: camera_buffer } }],
    });

    const gaussian_bind_group = device.createBindGroup({
        label: 'gaussian gaussians',
        layout: render_pipeline.getBindGroupLayout(1),
        entries: [
            { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
        ]
    });

    // ===============================================
    //    Command Encoder Functions
    // ===============================================
    const indirectDrawValues = new Uint32Array(4);
    indirectDrawValues[0] = 4; // vertex count
    indirectDrawValues[1] = pc.num_points; // instance count
    indirectDrawValues[2] = 0; // first vertex index
    indirectDrawValues[3] = 0; // first instance index

    const indirectDrawBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT,
    });
    device.queue.writeBuffer(indirectDrawBuffer, 0, indirectDrawValues, 0, indirectDrawValues.length);

    const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
        const pass = encoder.beginRenderPass({
            label: 'gaussian render',
            colorAttachments: [
                {
                    view: texture_view,
                    loadOp: 'clear',
                    storeOp: 'store',
                }
            ],
        });
        pass.setPipeline(render_pipeline);
        pass.setBindGroup(0, camera_bind_group);
        pass.setBindGroup(1, gaussian_bind_group);

        // pass.drawIndirect(indirectDrawBuffer, 0);
        pass.draw(4, pc.num_points, 0, 0);
        pass.end();
    };

    // ===============================================
    //    Return Render Object
    // ===============================================
    return {
        frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
            // sorter.sort(encoder);
            render(encoder, texture_view);
        },
        camera_buffer,
    };
}