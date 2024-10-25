import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
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

  const splat_buffer_size = pc.num_points * 8 * 3;
  const splat_buffer = createBuffer(device, 'splat buffer', splat_buffer_size, GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX);

  const splat_indirect_buffer_size = 4 * 4;
  const splat_indirect_buffer = createBuffer(device, 'splat indirect buffer', splat_indirect_buffer_size, GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE);
  device.queue.writeBuffer(splat_indirect_buffer, 0, new Uint32Array([4, 0, 0, 0]));

  const settings_buffer = createBuffer(
    device, 'settings buffer', 8,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    new Float32Array([1.0, pc.sh_deg,])
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

  const render_pipeline = device.createRenderPipeline({
    label: 'render pipeline',
    layout: 'auto',
    vertex: {
      entryPoint: 'vs_main',
      module: device.createShaderModule({code: renderWGSL})
    },
    fragment: {
      entryPoint: 'fs_main',
      module: device.createShaderModule({code: renderWGSL}),
      targets: [{
        format: presentation_format,
      }]
    },
});

  const render_bind_group = device.createBindGroup({
    label: 'render bind group',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[1].sort_indices_buffer } },
      { binding: 2, resource: { buffer: splat_buffer } }
    ]
  });

  // ===============================================
  //    Command Encoder Functions
  // ===============================================

  const preprocess = (encoder: GPUCommandEncoder) => {
    const preprocess_pass = encoder.beginComputePass();
    preprocess_pass.setPipeline(preprocess_pipeline);
    preprocess_pass.setBindGroup(0, camera_bind_group);
    preprocess_pass.setBindGroup(1, gaussian_bind_group);
    preprocess_pass.setBindGroup(2, sort_bind_group);
    preprocess_pass.setBindGroup(3, splat_bind_group);
    preprocess_pass.end();
  };

  const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    const render_pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: texture_view,
        loadOp: 'clear',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        storeOp: 'store',
      }],
    });

    render_pass.setPipeline(render_pipeline);
    render_pass.setBindGroup(0, render_bind_group);
    render_pass.drawIndirect(splat_indirect_buffer, 0);
    render_pass.end();
  };

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      preprocess(encoder);
      sorter.sort(encoder);
      encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, splat_indirect_buffer, 4, 4);
      render(encoder, texture_view);
    },
    camera_buffer,
  };
}
