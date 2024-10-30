import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter, c_histogram_block_rows, C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  render_settings_buffer: GPUBuffer,
}

// Utility function to create GPU buffers with optional data initialization
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

  // Initialize sorter for GPU-based sorting of point cloud data
  const sorter = get_sorter(pc.num_points, device);

  // Initialize GPU Buffers
  const nulling_data = new Uint32Array([0]);
  const null_buffer = createBuffer(device, 'null buffer', 4, GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, nulling_data);
  const splatBuffer = createBuffer(device, 'splat buffer', pc.num_points * 96, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const render_settings_buffer = createBuffer(device, 'render settings buffer', 4 * 2, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, new Float32Array([1.0, pc.sh_deg]));

  // Create Compute Pipeline and Bind Groups
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

  const preprocess_bind_group = device.createBindGroup({
    label: 'preprocess bind group',
    layout: preprocess_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
      { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
      { binding: 4, resource: { buffer: camera_buffer } },
      { binding: 5, resource: { buffer: pc.gaussian_3d_buffer } },
      { binding: 6, resource: { buffer: splatBuffer } },
      { binding: 7, resource: { buffer: render_settings_buffer } },
      { binding: 8, resource: { buffer: pc.sh_buffer } },
    ],
  });

  // Create Render Pipeline and Bind Groups
  const render_pipeline = device.createRenderPipeline({
    label: 'render gaussian',
    layout: 'auto',
    vertex: {
      module: device.createShaderModule({ code: renderWGSL }),
      entryPoint: 'vs_main'
    },
    fragment: {
      module: device.createShaderModule({ code: renderWGSL }),
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
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: splatBuffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 2, resource: { buffer: camera_buffer } },
    ],
  });

  const indirect_draw_data = new Uint32Array([6, 0, 0, 0]);
  const indirect_draw_buffer = createBuffer(device, 'indirect draw buffer', 4 * 4, GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST, indirect_draw_data);

  // Command Encoder Functions
  const preprocess = (encoder: GPUCommandEncoder) => {
    const preprocess_pass = encoder.beginComputePass();
    preprocess_pass.setPipeline(preprocess_pipeline);
    preprocess_pass.setBindGroup(0, preprocess_bind_group);
    preprocess_pass.dispatchWorkgroups(Math.ceil(pc.num_points / C.histogram_wg_size), 1, 1);
    preprocess_pass.end();
  };

  const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    const pass = encoder.beginRenderPass({
      label: "render gaussian pass",
      colorAttachments: [{
        view: texture_view,
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    pass.setPipeline(render_pipeline);
    pass.setBindGroup(0, render_bind_group);
    pass.drawIndirect(indirect_draw_buffer, 0);
    pass.end();
  };

  // Return Render Object with Frame Function
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      encoder.copyBufferToBuffer(null_buffer, 0, sorter.sort_info_buffer, 0, 4);
      encoder.copyBufferToBuffer(null_buffer, 0, sorter.sort_dispatch_indirect_buffer, 0, 4);

      preprocess(encoder);
      sorter.sort(encoder);

      encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, indirect_draw_buffer, 4, 4);

      render(encoder, texture_view);
    },
    camera_buffer,
    render_settings_buffer,
  };
}
