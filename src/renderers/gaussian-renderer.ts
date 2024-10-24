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
  
    // create the indirect buffer
    const indirect_buffer = createBuffer(
        device, 'indirect_buffer', 16,
        GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT,
        new Uint32Array([
            6,
            
            // render all the data for testing
            pc.num_points,
            
            0,
            0,
        ])
    );
    
    // create the render pipeline
    const render_pipeline = device.createRenderPipeline({
        label: 'render_pipeline',
        layout: 'auto',
        vertex: {
            module: device.createShaderModule({
                code: renderWGSL
            }),
            entryPoint: 'vs_main',
        },
        fragment: {
            module: device.createShaderModule({
                code: renderWGSL
            }),
            targets: [
                {
                    format: presentation_format,
                },
            ],
            entryPoint: 'fs_main',
        },
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
        
        // begin a new render pass
        const render_pass = encoder.beginRenderPass({
            label: 'render_pass',
            colorAttachments: [
                {
                    view: texture_view,
                    loadOp: 'clear',
                    storeOp: 'store',
                    clearValue: [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                },
            ],
        });
        
        // bind the render pipeline
        render_pass.setPipeline(render_pipeline);
        
        // perform drawing
        render_pass.drawIndirect(indirect_buffer, 0);
        
        // end the render pass
        render_pass.end();
    },
    camera_buffer,
  };
}
