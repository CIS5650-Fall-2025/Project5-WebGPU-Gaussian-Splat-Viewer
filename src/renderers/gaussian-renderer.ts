import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';
import { encode } from '@loaders.gl/core';

export interface GaussianRenderer extends Renderer {
    renderSettingBuffer:GPUBuffer;
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
 

  


  const indirectdraw_buffersize = 4 * Uint32Array.BYTES_PER_ELEMENT;
  const indirectdraw_buffer = device.createBuffer({
    label:"indirect draw buffer",
    size: indirectdraw_buffersize,
    usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })

  
  const indirectdraw_host = new Uint32Array([6, pc.num_points, 0, 0]);
  device.queue.writeBuffer(indirectdraw_buffer, 0, indirectdraw_host.buffer);


  const nulling_data = new Uint32Array([0]);
  const nulling_buffer = device.createBuffer({
    label: "indirect draw reset null buffer",
    size: 1 * Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  })

  device.queue.writeBuffer(nulling_buffer, 0, nulling_data.buffer);

  
  const splatData = new Uint32Array(pc.num_points * 6);


  const splatBuffer = createBuffer(
    device,
    'Splat Buffer',
    splatData.byteLength,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX,
  );

  const renderSettingsData = new Float32Array([1.0, pc.sh_deg]);
  const renderSettingBuffer = createBuffer(
    device,
    'Render setting Buffer',
    renderSettingsData.byteLength,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    renderSettingsData
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

  const computeBindGroup = device.createBindGroup({
    label: 'Compute Bind Group',
    layout: preprocess_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: splatBuffer } }, 
      { binding: 1, resource: { buffer: pc.sh_buffer } }, 
      { binding: 2, resource: { buffer: pc.gaussian_3d_buffer } }, 
    ],
  });

  const compute_setting_BindGroup = device.createBindGroup({
    label: 'Compute setting Bind Group',
    layout: preprocess_pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } }, 
      { binding: 1, resource: { buffer: renderSettingBuffer } }, 
     
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
  const renderPipeline = device.createRenderPipeline({
    label : "render pipeline",
    layout: "auto",
    vertex: {
        module: device.createShaderModule({
            label: "vert shader",
            code: renderWGSL,
        }),
        entryPoint: "vs_main",
        buffers: [],
    },
    fragment: {
        module: device.createShaderModule({
            label: "frag shader",
            code: renderWGSL,
        }),
        entryPoint: "fs_main",
        targets: [
          {
              format: presentation_format,
              blend: {
                  color: {
                    operation: 'add',
                    srcFactor: 'one',
                    dstFactor: 'one-minus-src-alpha',
                  },
                  alpha: {
                    operation: 'add',
                    srcFactor: 'one',
                    dstFactor: 'one-minus-src-alpha',
                  },
              },
          },
      ],
    }
  });
 
  const splatBindGroup = device.createBindGroup({
    label: 'Splat Bind Group',
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: splatBuffer } },
              { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
              { binding: 2, resource: { buffer: camera_buffer } }
      ]
    ,
    
  });



  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {

      encoder.copyBufferToBuffer(nulling_buffer, 0, sorter.sort_info_buffer, 0, 4)
      encoder.copyBufferToBuffer(nulling_buffer, 0, sorter.sort_dispatch_indirect_buffer, 0, 4)


      const computePass = encoder.beginComputePass({
        label: "Compute Pass",
      });
      computePass.setPipeline(preprocess_pipeline);
      computePass.setBindGroup(0, computeBindGroup);
      computePass.setBindGroup(1, compute_setting_BindGroup);
      computePass.setBindGroup(2, sort_bind_group);

      const workgroupSize = C.histogram_wg_size;
      const numWorkgroups = Math.ceil(pc.num_points / workgroupSize);

    
      computePass.dispatchWorkgroups(numWorkgroups);

      computePass.end();

      

      sorter.sort(encoder);

      encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, indirectdraw_buffer, 4, 4)


      const render_pass = encoder.beginRenderPass({
        label:"render pass",
        colorAttachments:[{
          view: texture_view,
          loadOp : "clear",
          storeOp: "store",
          clearValue: {r: 0, g:0, b:0, a:1},
        }]
      })
      render_pass.setPipeline(renderPipeline);
      render_pass.setBindGroup(0, splatBindGroup);
      //render_pass.setVertexBuffer(0, splatBuffer);
      render_pass.drawIndirect(indirectdraw_buffer, 0);
      render_pass.end();
    },
    camera_buffer,
    renderSettingBuffer,
  };
}
