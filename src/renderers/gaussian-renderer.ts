import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  render_setting_buffer: GPUBuffer
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
  const nulling_buffer = createBuffer(
    device,
    'nulling buffer',
    4,
    GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    nulling_data
  );

  const render_setting_buffer = createBuffer(
    device,
    'render setting buffer',
    8,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_SRC,
    new Float32Array([
      1.0,
      pc.sh_deg
    ])
  )

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

  const uniform_bind_group = device.createBindGroup({
    label: 'uniform_bind_group',
    layout: preprocess_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: camera_buffer }},
      { binding: 1, resource: { buffer: render_setting_buffer }},
    ],
  });

  const gaussian_bind_group = device.createBindGroup({
    label: 'gaussian_bind_group',
    layout: preprocess_pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: pc.gaussian_3d_buffer }},
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


  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================
  
  // Render pipeline
  const render_pipeline = device.createRenderPipeline({
    label: 'render pipeline',
    layout: 'auto',
    vertex: {
      module: device.createShaderModule({
        code: renderWGSL
      }),
      entryPoint: 'vs_main'
    },
    fragment: {
      module: device.createShaderModule({
        code: renderWGSL
      }),
      targets: [{
        format: presentation_format
      }],
      entryPoint: 'fs_main'
    }
  });

  const render_bind_group = device.createBindGroup({
    label: 'render bind group',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: camera_buffer }},
      { binding: 1, resource: { buffer: splat_buffer}},
      { binding: 2, resource: { buffer: sorter.ping_pong[0]}},
    ]
  })


  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  const preprocess = (encoder: GPUCommandEncoder) => {
    const preprocess_pass = encoder.beginComputePass();
    preprocess_pass.setPipeline(preprocess_pipeline);
    preprocess_pass.setBindGroup(0, uniform_bind_group);
    preprocess_pass.setBindGroup(1, gaussian_bind_group);
    preprocess_pass.setBindGroup(2, sort_bind_group);
    preprocess_pass.dispatchWorkgroups(Math.ceil(pc.num_points / C.histogram_wg_size));
    preprocess_pass.end();
  };

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      preprocess(encoder);
      sorter.sort(encoder);

      // Render pass
      const render_pass = encoder.beginRenderPass({
        label: 'gaussian render pass',
        colorAttachments: [{
          view: texture_view,
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: [0, 0, 0, 1]
        }]
      });

      render_pass.setPipeline(render_pipeline);
      
      render_pass.end();
    },
    camera_buffer,
    render_setting_buffer
  };
}
