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
  const f32_size = 4;
  const splat_size = 3 * f32_size; // x,y,z at f32
  const num_points = pc.num_points;
  const guassian_splat_buffer = createBuffer(
    device, 'gaussian splat buffer', 
    num_points * splat_size, GPUBufferUsage.STORAGE);

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================

  // Preprocess Bind Group Layouts
  const preprocess_projection_bind_group_layout = device.createBindGroupLayout({
    label: 'preprocess projection bind group layout',
    entries: [
      {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }},
      {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' }}
    ],
  }); 
  const gs_preprocess_bind_group_layout = device.createBindGroupLayout({
    label: 'gaussian preprocess bind group layout',
    entries: [
      {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' }}
    ],
  });
  const sort_bind_group_layout = device.createBindGroupLayout({
    label: 'sort bind group layout',
    entries: [
      {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' }},
      {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' }},
      {binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' }},
      {binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' }}
    ],
  });

  const preprocess_pipeline = device.createComputePipeline({
    label: 'preprocess',
    layout: device.createPipelineLayout({
      label: 'preprocess pipeline layout',
      bindGroupLayouts: [
        preprocess_projection_bind_group_layout, 
        gs_preprocess_bind_group_layout, 
        sort_bind_group_layout
      ]
    }),
    compute: {
      module: device.createShaderModule({ code: preprocessWGSL }),
      entryPoint: 'preprocess',
      constants: {
        workgroupSize: C.histogram_wg_size,
        sortKeyPerThread: c_histogram_block_rows,
      },
    },
  });

  const preprocess_projection_bind_group = device.createBindGroup({
    label: 'preprocess projection',
    layout: preprocess_projection_bind_group_layout,
    entries: [
      {binding: 0, resource: { buffer: camera_buffer }},
      {binding: 1, resource: { buffer: pc.gaussian_3d_buffer }},
    ],
  });

  const gs_preprocess_bind_group = device.createBindGroup({
    label: 'gaussian',
    layout: gs_preprocess_bind_group_layout,
    entries: [
      {binding: 0, resource: { buffer: guassian_splat_buffer }},
    ],
  });

  const sort_bind_group = device.createBindGroup({
    label: 'sort',
    layout: sort_bind_group_layout,
    entries: [
      { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
      { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
    ],
  });

  const preprocess_compute = (encoder: GPUCommandEncoder) => {
    const pass = encoder.beginComputePass();
    pass.setPipeline(preprocess_pipeline);
    pass.setBindGroup(0, preprocess_projection_bind_group);
    pass.setBindGroup(1, gs_preprocess_bind_group);
    pass.setBindGroup(2, sort_bind_group);
    pass.dispatchWorkgroups(pc.num_points / C.histogram_wg_size);
    pass.end();
  };

  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================
  const gs_render_bind_group_layout = device.createBindGroupLayout({
    label: 'gaussian render bind group layout',
    entries: [
      {binding: 0, 
       visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, 
       buffer: { type: 'read-only-storage' }}
    ],
  });

  const gs_render_bind_group = device.createBindGroup({
    label: 'gaussian render bind group ',
    layout: gs_render_bind_group_layout,
    entries: [
      {binding: 0, resource: { buffer: guassian_splat_buffer }},
    ],
  });

  const render_shader = device.createShaderModule({ code: renderWGSL });
  const gaussian_render_pipeline = device.createRenderPipeline({
    label: 'gaussian render',
    layout: device.createPipelineLayout({
      label: 'gaussian render pipeline layout',
      bindGroupLayouts: [gs_render_bind_group_layout]
    }),
    vertex: {
      module: render_shader,
      entryPoint: 'vs_main',
    },
    fragment: {
      module: render_shader,
      entryPoint: 'fs_main',
      targets: [{ format: presentation_format }],
    },
    primitive: {
      topology: 'triangle-list',
    },
  });

  const gaussian_render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    const pass = encoder.beginRenderPass({
      label: 'gaussian render',
      colorAttachments: [
        { 
          view: texture_view, 
          loadOp: 'clear', 
          storeOp: 'store' 
        }
      ],
    });
    pass.setPipeline(gaussian_render_pipeline);
    pass.setBindGroup(0, gs_render_bind_group);
    pass.draw(6, num_points); // 6 vertices per quad
    pass.end();
  }

  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      preprocess_compute(encoder);
      sorter.sort(encoder);
      gaussian_render(encoder, texture_view);
    },
    camera_buffer,
  };
}
