import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  settingsBuffer : GPUBuffer;
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
  const nullBuffer  = createBuffer(
    device,
    'Clear Buffer ',
    4,
    GPUBufferUsage.COPY_SRC,
    nulling_data);

  const settingsBuffer  = createBuffer(
      device,
      'Render Settings Buffer ',
      8, 
      GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
      new Float32Array([1.0, pc.sh_deg])
  );

  const splatBuffer = createBuffer(
    device,
    'splatBuffer',
    (8 + 8 + 8) * pc.num_points, // Position and size, color, conic and opacity
    GPUBufferUsage.STORAGE,
    null
  );

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

  const camera_bind_group = device.createBindGroup({
    label: 'point cloud camera',
    layout: preprocess_pipeline.getBindGroupLayout(0),
    entries: [{binding: 0, resource: { buffer: camera_buffer }}],
  });

  const gaussian_bind_group = device.createBindGroup({
    label: 'gaussian_bind_group',
    layout: preprocess_pipeline.getBindGroupLayout(1),
    entries: [{ binding: 0, resource: { buffer: pc.gaussian_3d_buffer }}],
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
  const buffer = createBuffer(
    device, //device
    "Buffer", //label
    16, //size
    GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT, // Usage
    new Uint32Array([6, pc.num_points, 0, 0, ]) // Data
  )
  const render_shader = device.createShaderModule({code: renderWGSL});
  const render_pipeline = device.createRenderPipeline({
    label: 'render',
    layout: 'auto',
    vertex: {
      module: render_shader,
      entryPoint: 'vs_main',
    },
    fragment: {
      module: render_shader,
      entryPoint: 'fs_main',
      targets: [{ format: presentation_format,
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
    },
  });

  const preprocessor_bind_group = device.createBindGroup({
    label: 'preprocessor_bind_group',
    layout: preprocess_pipeline.getBindGroupLayout(3),
    entries: [
      { binding: 0, resource: { buffer: splatBuffer } },
      { binding: 1, resource: { buffer: settingsBuffer  } },
      { binding: 2, resource: { buffer: pc.sh_buffer } },
    ],
  });

  const gaussian_render_bind_group = device.createBindGroup({
      label: 'gaussian_render_bind_group',
      layout: render_pipeline.getBindGroupLayout(0),
      entries: [
          { binding: 0, resource: { buffer: splatBuffer } },
          { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
          { binding: 2, resource: { buffer: camera_buffer } }, 
      ],
  });

  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  const compute = (encoder: GPUCommandEncoder) => {
      const compute_pass = encoder.beginComputePass();
      compute_pass.setPipeline(preprocess_pipeline);
      compute_pass.setBindGroup(0, camera_bind_group);
      compute_pass.setBindGroup(1, gaussian_bind_group);
      compute_pass.setBindGroup(2, sort_bind_group);
      compute_pass.setBindGroup(3, preprocessor_bind_group);
      compute_pass.dispatchWorkgroups(Math.ceil(pc.num_points / C.histogram_wg_size));
      compute_pass.end();
  };

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      encoder.copyBufferToBuffer(nullBuffer , 0, sorter.sort_info_buffer, 0, 4);
      encoder.copyBufferToBuffer(nullBuffer , 0, sorter.sort_dispatch_indirect_buffer, 0, 4);
      compute(encoder);
      sorter.sort(encoder);
      encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, buffer, 4, 4);
      const pass = encoder.beginRenderPass({
          label: 'Render  pass',
          colorAttachments: [
            {
              view: texture_view,
              loadOp: 'clear',
              storeOp: 'store',
              clearValue: [0.0, 0.0, 0.0, 1.0],
          }
        ],
      });
      pass.setPipeline(render_pipeline);
      pass.setBindGroup(0, gaussian_render_bind_group);
      pass.drawIndirect(buffer, 0); 
      pass.end();
    },
    camera_buffer,
    settingsBuffer ,
  };
}
