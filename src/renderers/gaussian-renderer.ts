import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';
import { vec2 } from 'wgpu-matrix';

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

  const render_settings_buffer = createBuffer(
    device,
    "render settings buffer",
    8,
    GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
    vec2.create(1.0, pc.sh_deg)
  );

  const SPLAT_SIZE = 16;
  const splat_buffer = createBuffer(
    device,
    "splat buffer",
    pc.num_points * SPLAT_SIZE,
    GPUBufferUsage.STORAGE,
    null,
  );

  const nulling_data = new Uint32Array([0]);

  const nulling_data_buffer = createBuffer(
    device,
    "nulling data buffer",
    4,
    GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    nulling_data
  );

  const indirect_buffer = createBuffer(
    device,
    "indirect buffer",
    16,
    GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT,
    new Uint32Array([6, pc.num_points, 0, 0])
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
    label: 'camera bind group',
    layout: preprocess_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },
      { binding: 1, resource: { buffer: render_settings_buffer } }
    ],
  });

  const gaussian_bind_group = device.createBindGroup({
    label: 'gaussians',
    layout: preprocess_pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
    ],
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

  const splat_bind_group = device.createBindGroup({
    label: 'splat bind group',
    layout: preprocess_pipeline.getBindGroupLayout(3),
    entries: [
      { binding: 0, resource: { buffer: splat_buffer } }
    ]
  })

  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================

  const render_shader = device.createShaderModule({code: renderWGSL})
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
      targets: [{ format: presentation_format }],
    },
  });

  const render_bind_group = device.createBindGroup({
    label: 'render bind group',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer }},
      { binding: 1, resource: { buffer: splat_buffer }}
    ]
  })


  // ===============================================
  //    Command Encoder Functions
  // ===============================================

  const preprocess = (encoder: GPUCommandEncoder) => {
    const compute_pass = encoder.beginComputePass({ label: "preprocess gaussians" })
    compute_pass.setPipeline(preprocess_pipeline);
    compute_pass.setBindGroup(0, camera_bind_group);
    compute_pass.setBindGroup(1, gaussian_bind_group);
    compute_pass.setBindGroup(2, sort_bind_group);
    compute_pass.setBindGroup(3, splat_bind_group);
    compute_pass.dispatchWorkgroups(Math.ceil(pc.num_points / C.histogram_wg_size));
    compute_pass.end();
  };

  const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {

    // encoder.copyBufferToBuffer(nulling_data_buffer, 0, sorter.sort_info_buffer, 0, 4);
    // encoder.copyBufferToBuffer(nulling_data_buffer, 0, sorter.sort_dispatch_indirect_buffer, 0, 4);
    // encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, indirect_buffer, 4, 4);

    const pass = encoder.beginRenderPass({
      label: 'gaussian render',
      colorAttachments: [{
          view: texture_view,
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: [0.0, 0.0, 0.0, 1.0]
        }
      ],
    });
    pass.setPipeline(render_pipeline);
    // pass.setBindGroup(0, camera_bind_group);
    pass.setBindGroup(0, render_bind_group);
    pass.drawIndirect(indirect_buffer, 0);
    pass.end();
  }

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      preprocess(encoder);
      // sorter.sort(encoder);
      render(encoder, texture_view);
    },
    camera_buffer,
  };
}
