import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  render_settings_buffer: GPUBuffer;
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



  const ind_draw_data = new Uint32Array([4, pc.num_points, 0, 0]);

  const zero_data = new Uint32Array([0]);

  const zero_buffer = createBuffer(
    device, 'zero_buffer', 4, GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 
    zero_data);

  const ind_draw_buf = createBuffer(
    device, 
    'ind_draw_buf',
    16,
    GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
  );

  device.queue.writeBuffer(ind_draw_buf, 0, ind_draw_data);

  //render settings
  const renderSettings = new Float32Array([1.0, pc.sh_deg]); // Example values for gaussian_scaling and sh_deg

  const render_settings_buffer = createBuffer(
    device,
    'render_settings',
    renderSettings.byteLength,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    renderSettings
  );

  //splat buffer
  const splat_buffer = device.createBuffer({
    label: 'splat buffer',
    size: pc.num_points * 7 * 4, // Adjust size based on what data you store per splat
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================


  /*
  const sort_bind_group_layout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  */


  /*
  const preprocess_bind_group_layout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  */

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
      { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
      { binding: 1, resource: { buffer: splat_buffer } }, // splat buffer for writing
      { binding: 2, resource: { buffer: camera_buffer } },
      { binding: 3, resource: { buffer: render_settings_buffer} },
      { binding: 4, resource: { buffer: pc.sh_buffer } },
    ],
  });

  const sort_bind_group = device.createBindGroup({
    label: 'sort',
    layout: preprocess_pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
      { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
    ],
  });

  /*
  const preprocess_pipeline_layout = device.createPipelineLayout({
    bindGroupLayouts: [
      preprocess_bind_group_layout, // This is for group(0)
      sort_bind_group_layout        // This is for group(1)
    ],
  });
  */


  //enable error logging
  device.onuncapturederror = (error) => {
    console.error("WebGPU Error:", error);
  };


  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================
  /*
  const render_bind_group_layout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    ],
  });
  */

  const render_shader = device.createShaderModule({code: renderWGSL});

  const render_pipeline = device.createRenderPipeline({
    label: 'render',
    layout: 'auto',
    vertex: {
      module: render_shader,
      entryPoint: 'vs_main',
      buffers: [],
    },
    fragment: {
      module: render_shader,
      entryPoint: 'fs_main',
      targets: [{ 
        format: presentation_format,
        blend: {
          color: {
            srcFactor: 'src-alpha',
            dstFactor: 'one-minus-src-alpha',
            operation: 'add',
          },
          alpha: {
            srcFactor: 'one',
            dstFactor: 'one-minus-src-alpha',
            operation: 'add',
          }
        }

       }],
    },
    primitive: {
      topology: 'triangle-strip',
    },
  });

  const render_bind_group = device.createBindGroup({
    label: 'render bind group',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },
      { binding: 1, resource: { buffer: splat_buffer } }, // splat buffer for reading
      { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
    ],
  });

  /*
  const render_pipeline_layout = device.createPipelineLayout({
    bindGroupLayouts: [
      render_bind_group_layout,
    ],
  });
  */
  


  /*
  const camera_bind_group = device.createBindGroup({
    label: 'point cloud camera',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [{binding: 0, resource: { buffer: camera_buffer }}],
  });

  const gaussian_bind_group = device.createBindGroup({
    label: 'gaussians',
    layout: render_pipeline.getBindGroupLayout(1),
    entries: [
      {binding: 0, resource: { buffer: pc.gaussian_3d_buffer }},
    ],
  });
  */


  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  
  const preprocess = (encoder: GPUCommandEncoder) => {
    const pass = encoder.beginComputePass();
    pass.setPipeline(preprocess_pipeline);
    pass.setBindGroup(0, preprocess_bind_group);
    pass.setBindGroup(1, sort_bind_group);
    pass.dispatchWorkgroups(Math.ceil(pc.num_points / C.histogram_wg_size));
    pass.end();
  };
  
  
  const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    const pass = encoder.beginRenderPass({
      label: 'point cloud render',
      colorAttachments: [
        {
          view: texture_view,
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
        }
      ],
    });
    pass.setPipeline(render_pipeline);
    pass.setBindGroup(0, render_bind_group);
    pass.drawIndirect(ind_draw_buf, 0);
    pass.end();
  };
  

  // ===============================================
  //    Return Render Object
  // ===============================================
  //must also return render settings buffer, err elsewise
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      encoder.copyBufferToBuffer(zero_buffer, 0, sorter.sort_dispatch_indirect_buffer, 0, 4);
      encoder.copyBufferToBuffer(zero_buffer, 0, sorter.sort_info_buffer, 0, 4);
      preprocess(encoder);
      sorter.sort(encoder);
      encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, ind_draw_buf, 4, 4);
      render(encoder, texture_view);
    },
    camera_buffer,
    render_settings_buffer,
  };
}
