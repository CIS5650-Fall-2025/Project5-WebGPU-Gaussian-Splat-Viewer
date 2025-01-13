import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter, c_histogram_block_rows, C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
    renderSettingsBuffer: GPUBuffer;
};

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

    const workgroupSize = 64;

    // ===============================================
    //            Initialize GPU Buffers
    // ===============================================
    const nullingDataBuffer = createBuffer(
        device,
        'nulling data buffer',
        4,
        GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        new Uint32Array([0])
    );

    const renderSettingsBuffer = createBuffer(
        device,
        'render settings buffer',
        8,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        new Float32Array([1.0, pc.sh_deg])
    );

    const splatSize = 4 * 12;
    const splatBuffer = createBuffer(
        device,
        "splat buffer",
        splatSize * pc.num_points,
        GPUBufferUsage.STORAGE
    );

    const indirectDrawBuffer = createBuffer(
        device,
        'indirect draw buffer',
        4 * 4,
        GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
        new Uint32Array([4, 0, 0, 0])
    );
    

    // ===============================================
    //    Create Compute Pipeline and Bind Groups
    // ===============================================
    const preprocessPipeline = device.createComputePipeline({
        label: 'preprocess',
        layout: 'auto',
        compute: {
            module: device.createShaderModule({ code: preprocessWGSL }),
            entryPoint: 'preprocess',
            constants: {
                workgroupSize: C.histogram_wg_size,
                sortKeyPerThread: c_histogram_block_rows
            },
        },
    });

    const uniformsBindGroup = device.createBindGroup({
        label: 'uniforms',
        layout: preprocessPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: camera_buffer } },
            { binding: 1, resource: { buffer: renderSettingsBuffer } },
        ]
    });

    const gaussiansBindGroup = device.createBindGroup({
        label: 'gaussians',
        layout: preprocessPipeline.getBindGroupLayout(1),
        entries: [
            { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
            { binding: 1, resource: { buffer: pc.sh_buffer } },
            { binding: 2, resource: { buffer: splatBuffer } },
        ],
    });

    const sortBindGroup = device.createBindGroup({
        label: 'sort',
        layout: preprocessPipeline.getBindGroupLayout(2),
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
    const renderShader = device.createShaderModule({
        code: renderWGSL,
        label: 'gaussian render shader'
    });

    const renderPipeline = device.createRenderPipeline({
        label: 'gaussian render pipeline',
        layout: 'auto',
        vertex: {
            module: renderShader,
            entryPoint: 'vs_main'
        },
        fragment: {
            module: renderShader,
            entryPoint: 'fs_main',
            targets: [
                { 
                    format: presentation_format,
                    blend: {
                        color: {
                            operation: 'add',
                            srcFactor: 'one',
                            dstFactor: 'one-minus-src-alpha',
                        },
                        alpha: {
                            operation: 'add',
                            srcFactor: 'one',
                            dstFactor: 'one-minus-src-alpha',
                        },
                      }
                }
            ],
        },
        primitive: {
            topology: "triangle-strip"
        }
    });

    const cameraBindGroup = device.createBindGroup({
        label: 'camera',
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: camera_buffer } }],
    });

    const splatsBindGroup = device.createBindGroup({
        label: 'gaussian gaussians',
        layout: renderPipeline.getBindGroupLayout(1),
        entries: [
            { binding: 0, resource: { buffer: splatBuffer } },
            { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
        ]
    });

    // ===============================================
    //    Command Encoder Functions
    // ===============================================
    const preprocess = (encoder: GPUCommandEncoder) => {
        // initialize two atomic add variables to zero
        encoder.copyBufferToBuffer(nullingDataBuffer, 0, sorter.sort_info_buffer, 0, 4);
        encoder.copyBufferToBuffer(nullingDataBuffer, 0, sorter.sort_dispatch_indirect_buffer, 0, 4);
        
        const computePass = encoder.beginComputePass({
            label: 'preprocess',
        });
        computePass.setPipeline(preprocessPipeline);
        computePass.setBindGroup(0, uniformsBindGroup);
        computePass.setBindGroup(1, gaussiansBindGroup);
        computePass.setBindGroup(2, sortBindGroup);
        computePass.dispatchWorkgroups(
            Math.ceil(pc.num_points / workgroupSize)
        );
        computePass.end();

        // copy keys_size into indirect draw buffer
        encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, indirectDrawBuffer, 4, 4);
    }

    const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
        const renderPass = encoder.beginRenderPass({
            label: 'gaussian render',
            colorAttachments: [
                {
                    view: texture_view,
                    loadOp: 'clear',
                    storeOp: 'store'
                }
            ],
        });
        renderPass.setPipeline(renderPipeline);
        renderPass.setBindGroup(0, cameraBindGroup);
        renderPass.setBindGroup(1, splatsBindGroup);
        renderPass.drawIndirect(indirectDrawBuffer, 0);
        renderPass.end();
    };

    // ===============================================
    //    Return Render Object
    // ===============================================
    return {
        frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
            preprocess(encoder);
            sorter.sort(encoder);
            render(encoder, texture_view);
        },
        camera_buffer,
        renderSettingsBuffer
    };
}