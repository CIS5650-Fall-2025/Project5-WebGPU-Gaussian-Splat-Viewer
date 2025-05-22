import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  rendering_buffer: GPUBuffer;
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
    Uint32Array.BYTES_PER_ELEMENT,
    GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    nulling_data
  );

  const indirect_buffer = createBuffer(
    device,
    'indirect buffer',
    20,
    GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT,
    new Uint32Array([6, 0, 0, 0, 0])
  );

  const rendering_buffer = createBuffer(
    device, 
    'render settings buffer', 
    Float32Array.BYTES_PER_ELEMENT,
    GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM, 
    new Float32Array([1.0])
  );

  const floatsPerSplat = 24;
  const splat_buffer = createBuffer(
    device,
    'gauss splat buffer',
    floatsPerSplat * pc.num_points,
    GPUBufferUsage.STORAGE
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
        shDegree: pc.sh_deg,
      },
    },
  });

  const [cameraLayout, gaussianLayout, sortingLayout, splatLayout] = [
  preprocess_pipeline.getBindGroupLayout(0),
  preprocess_pipeline.getBindGroupLayout(1),
  preprocess_pipeline.getBindGroupLayout(2),
  preprocess_pipeline.getBindGroupLayout(3),
];

  const sort_bind_group = device.createBindGroup({
    label: 'sort',
    layout: sortingLayout,
    entries: [
      { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
      { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
    ],
  });

  const camera_bind_group = device.createBindGroup({
    label: 'preprocess: camera bind group',
    layout: cameraLayout,
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } }
    ]
  });

  const gaussian_bind_group = device.createBindGroup({
    label: 'preprocess: gaussian bind group',
    layout: gaussianLayout,
    entries: [
      { binding: 0, resource: { buffer: pc.gaussian_3d_buffer }},
    ],
  });

  const splat_bind_group = device.createBindGroup({
    label: 'preprocess splat bind group',
    layout: splatLayout,
    entries: [
      {  binding: 0, resource: { buffer: splat_buffer } }, 
      { binding: 1, resource: { buffer: rendering_buffer } },
      { binding: 2, resource: { buffer: pc.sh_buffer } }
    ]
  });

  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================

  const render_pipeline = device.createRenderPipeline({
    label: 'gauss:render-pipeline',
    layout: 'auto',
    vertex: {
      module: device.createShaderModule({ code: renderWGSL }),
      entryPoint: 'vs_main',
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
              operation: 'add'
          },
          alpha: {
              srcFactor: 'one',
              dstFactor: 'one-minus-src-alpha',
              operation: 'add'
          }
        }
      }],
    },
  });

  const render_bind_group = device.createBindGroup({
  label: 'gauss: render bind group',
  layout: render_pipeline.getBindGroupLayout(0),
  entries: [
    {
      binding: 0,
      resource: { buffer: splat_buffer },
    },
    { 
      binding: 1,
      resource: { buffer: sorter.ping_pong[0].sort_indices_buffer },
    },
    {
      binding: 2,
      resource: {buffer: camera_buffer}
    }
  ],
});

  // ===============================================
  //    Command Encoder Functions
  // ===============================================

  const compute_pass = (encoder: GPUCommandEncoder) => {
    const preprocess_compute_pass = encoder.beginComputePass()
    preprocess_compute_pass.setPipeline(preprocess_pipeline);
    preprocess_compute_pass.setBindGroup(0, camera_bind_group);
    preprocess_compute_pass.setBindGroup(1, gaussian_bind_group);
    preprocess_compute_pass.setBindGroup(2, sort_bind_group);
    preprocess_compute_pass.setBindGroup(3, splat_bind_group);
    preprocess_compute_pass.dispatchWorkgroups(Math.ceil(pc.num_points / C.histogram_wg_size));
    preprocess_compute_pass.end();
  };

  const render_pass = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    const gaussian_render_pass = encoder.beginRenderPass({
      label: 'gaussian render pass',
      colorAttachments: [{
          view: texture_view,
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: [0, 0, 0, 1],
        }
      ],
    });
    gaussian_render_pass.setPipeline(render_pipeline);
    gaussian_render_pass.setBindGroup(0, render_bind_group);
    gaussian_render_pass.drawIndirect(indirect_buffer, 0);
    gaussian_render_pass.end();
  }


  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {

      encoder.copyBufferToBuffer(nulling_buffer, 0, sorter.sort_info_buffer, 0, 4);

      encoder.copyBufferToBuffer(nulling_buffer, 0, sorter.sort_dispatch_indirect_buffer, 0, 4);
      compute_pass(encoder);
      sorter.sort(encoder);
      encoder.copyBufferToBuffer(
        sorter.sort_info_buffer, 0, indirect_buffer, 4, 4);
      render_pass(encoder, texture_view);
    },
    camera_buffer,
    rendering_buffer
  };
}
