import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  scaling_buffer: GPUBuffer
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

  const scaling_buffer = createBuffer(
    device, 
    'render_settings_buffer', 
    4,
    GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM, 
    new Float32Array([1.0])
  );
  
  const nulling_data = new Uint32Array([0]);

  const nulling_buffer = createBuffer(
    device, 
    'nulling buffer', 
    4,
    GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 
    nulling_data
  );

  const splat_size = 8;

  const splat_buffer = createBuffer(
    device, 
    'splat buffer', 
    splat_size * pc.num_points, 
    GPUBufferUsage.STORAGE, 
    null
  );

  const indirect_buffer = createBuffer(
    device, 
    'indirect buffer', 
    20, 
    GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT,
    new Uint32Array([6, pc.num_points, 0, 0, 0])
  );

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================

  const preprocess_pipeline = device.createComputePipeline({
    label: 'preprocess pipeline',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: preprocessWGSL }),
      entryPoint: 'preprocess',
      constants: {
        workgroupSize: C.histogram_wg_size,
        sortKeyPerThread: c_histogram_block_rows
      }
    }
  });

  const camera_bind_group = device.createBindGroup({
    label: 'preprocess camera bind group',
    layout: preprocess_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } }
    ]
  });

  const gaussian_bind_group = device.createBindGroup({
    label: 'preprocess gaussians bind group',
    layout: preprocess_pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } }
    ]
  });

  const sort_bind_group = device.createBindGroup({
    label: 'preprocess sort bind group',
    layout: preprocess_pipeline.getBindGroupLayout(2),
    entries: [
      { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
      { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } }
    ]
  });

  const splat_preprocess_bind_group = device.createBindGroup({
    label: 'preprocess splat bind group',
    layout: preprocess_pipeline.getBindGroupLayout(3),
    entries: [
      { binding: 0, resource: { buffer: splat_buffer } },
      { binding: 1, resource: { buffer: scaling_buffer }}
    ]
  });

  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================

  const gaussian_render_pipeline = device.createRenderPipeline({
    label: 'gaussian render pipeline',
    layout: 'auto',
    vertex: {
      module: device.createShaderModule({ code: renderWGSL }),
      entryPoint: 'vs_main'
    },
    fragment: {
      module: device.createShaderModule({ code: renderWGSL }),
      entryPoint: 'fs_main',
      targets: [{ format: presentation_format }]
    }
  });

  const splat_render_bind_group = device.createBindGroup({
    label: 'gaussian render splat bind group',
    layout: gaussian_render_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: splat_buffer } }
    ]
  });

  // ===============================================
  //    Command Encoder Functions
  // ===============================================

  const preprocess_pass = (encoder: GPUCommandEncoder) => {
    const preprocess_pass = encoder.beginComputePass();

    preprocess_pass.setPipeline(preprocess_pipeline);

    preprocess_pass.setBindGroup(0, camera_bind_group);
    preprocess_pass.setBindGroup(1, gaussian_bind_group);
    preprocess_pass.setBindGroup(2, sort_bind_group);
    preprocess_pass.setBindGroup(3, splat_preprocess_bind_group);
    
    preprocess_pass.dispatchWorkgroups(Math.ceil(pc.num_points / C.histogram_wg_size));
    
    preprocess_pass.end();
  };
  
  const gaussian_render_pass = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    const gaussian_pass = encoder.beginRenderPass({
      label: 'gaussian render',
      colorAttachments: [
        {
          view: texture_view,
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: [0, 0, 0, 1]
        }
      ],
    });
    gaussian_pass.setPipeline(gaussian_render_pipeline);
    gaussian_pass.setBindGroup(0, splat_render_bind_group);

    gaussian_pass.drawIndirect(indirect_buffer, 0);
    gaussian_pass.end();
  };

  // ===============================================
  //    Return Render Object
  // ===============================================

  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {

      encoder.copyBufferToBuffer(
        nulling_buffer, 
        0,
        sorter.sort_info_buffer, 
        0,
        4
      );

      encoder.copyBufferToBuffer(
        nulling_buffer, 
        0,
        sorter.sort_dispatch_indirect_buffer, 
        0,
        4
      );

      preprocess_pass(encoder);

      //sorter.sort(encoder);

      encoder.copyBufferToBuffer(
        sorter.sort_info_buffer, 
        0,
        indirect_buffer, 
        4,
        4
      );

      gaussian_render_pass(encoder, texture_view);

    },

    camera_buffer,
    scaling_buffer
  };
}
