import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';
import { encode } from '@loaders.gl/core';

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


  const indirectdraw_buffersize = 4 * Uint32Array.BYTES_PER_ELEMENT;
  const indirectdraw_buffer = device.createBuffer({
    label:"indirect draw buffer",
    size: indirectdraw_buffersize,
    usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })

  const indirectdraw_host = new Uint32Array([6, pc.num_points, 0, 0]);
  device.queue.writeBuffer(indirectdraw_buffer, 0, indirectdraw_host.buffer);

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
  this.renderPipeline = device.createRenderPipeline({
    label : "render pipeline",
    layout: "auto",
    vertex: {
        module: device.createShaderModule({
            label: "vert shader",
            code: renderWGSL,
        }),
        entryPoint: "vs_main",
        buffers: [{
          arrayStride: 2 * Float32Array.BYTES_PER_ELEMENT,
          stepMode: "vertex",
          attributes:[{
            shaderLocation:0,
            offset:0,
            format: 'float32x2'

          }],
        }],
    },
    fragment: {
        module: device.createShaderModule({
            label: "frag shader",
            code: renderWGSL,
        }),
        entryPoint: "fs_main",
        targets: [{format: presentation_format}]
    }
});



  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      sorter.sort(encoder);
      const render_pass = encoder.beginRenderPass({
        label:"render pass",
        colorAttachments:[{
          view: texture_view,
          loadOp : "clear",
          storeOp: "store",
          clearValue: {r: 0, g:0, b:0, a:1},
        }]
      })
      render_pass.end();
    },
    camera_buffer,
  };
}
