import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  settings_buffer:GPUBuffer
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
  const nulling_buffer = createBuffer(device,
                                      'null buffer',
                                      4,
                                      GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
                                      nulling_data);

  const splat_buffer = createBuffer(device,
                                    'splat buffer',
                                    pc.num_points*24,
                                    GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST|GPUBufferUsage.STORAGE,
                                    null);

  const settings_buffer = createBuffer(device,
                                      'settings buffer',
                                      8,
                                      GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
                                      new Float32Array([1.0, pc.sh_deg]));                       

  const drawval = new Uint32Array(4);
                                      
  drawval[0] = 6; // The vertexCount value
  drawval[1] = pc.num_points; // The instanceCount value
  drawval[2] = 0; // The firstVertex value
  drawval[3] = 0; // The firstInstance value
  const drawbuffer = createBuffer(device,
                                  'draw buffer',
                                  16,
                                  GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT,
                                  drawval

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
  
  const preprocess_bind_group = device.createBindGroup({
    label: 'preprocess',
    layout: preprocess_pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
      { binding: 1, resource: { buffer:  settings_buffer} },
      { binding: 2, resource: { buffer:  pc.sh_buffer} },
      { binding: 3, resource: { buffer:  splat_buffer} }
    ],
  });

  const camera_bind_group = device.createBindGroup({
    label: 'camera',
    layout: preprocess_pipeline.getBindGroupLayout(0),
    entries:[{
      binding: 0, resource: {buffer: camera_buffer}
    }]
  });

  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  
  const compute_shader = (encoder: GPUCommandEncoder) => {
    encoder.copyBufferToBuffer(nulling_buffer,0,sorter.sort_info_buffer,0,4);
    encoder.copyBufferToBuffer(nulling_buffer,0,sorter.sort_dispatch_indirect_buffer,0,4);

    const pass = encoder.beginComputePass({ label: 'compute preprocess pass' });   
    pass.setPipeline(preprocess_pipeline);
    pass.setBindGroup(0, camera_bind_group);
    pass.setBindGroup(1, preprocess_bind_group);
    pass.setBindGroup(2, sort_bind_group);
    pass.dispatchWorkgroups(pc.num_points/C.histogram_wg_size);
    pass.end();
    
  }; 

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

  const render_bind_group = device.createBindGroup({
    label: 'preprocess',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },
      { binding: 1, resource: { buffer:  splat_buffer} },
      { binding: 2, resource: { buffer:  sorter.ping_pong[0].sort_indices_buffer} }
    ],
  });
  const render = (encoder: GPUCommandEncoder,texture_view: GPUTextureView) =>{
    const pass = encoder.beginRenderPass({
      label: 'point cloud render',
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
    pass.setBindGroup(0, render_bind_group);

    pass.drawIndirect(drawbuffer,0);
    pass.end();
  };
  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      compute_shader(encoder);
      sorter.sort(encoder);
      encoder.copyBufferToBuffer(sorter.sort_info_buffer,0,drawbuffer,4,4);
      render(encoder,texture_view);
    },
    camera_buffer,
    settings_buffer,
  };
}
