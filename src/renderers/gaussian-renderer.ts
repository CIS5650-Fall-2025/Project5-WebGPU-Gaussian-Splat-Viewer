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
  // const preprocess_pipeline = device.createComputePipeline({
  //   label: 'preprocess',
  //   layout: 'auto',
  //   compute: {
  //     module: device.createShaderModule({ code: preprocessWGSL }),
  //     entryPoint: 'preprocess',
  //     constants: {
  //       workgroupSize: C.histogram_wg_size,
  //       sortKeyPerThread: c_histogram_block_rows,
  //     },
  //   },
  // });

  // const sort_bind_group = device.createBindGroup({
  //   label: 'sort',
  //   layout: preprocess_pipeline.getBindGroupLayout(2),
  //   entries: [
  //     { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
  //     { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
  //     { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
  //     { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
  //   ],
  // });


  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================
  const render_pipeline = device.createRenderPipeline({
    label: 'render gaussian',
    layout: 'auto',
    vertex: {
      module: device.createShaderModule({ code: renderWGSL }),
      entryPoint: 'vs_main',
      buffers: [
        {
          arrayStride: 3 * 4,
          attributes: [
            {
              shaderLocation: 0,
              offset: 0,
              format: 'float32x3',
            },
          ],
        },
      ],
    },
    fragment: {
      module: device.createShaderModule({ code: renderWGSL }),
      entryPoint: 'fs_main',
      targets: [{ format: presentation_format }],
    },
    primitive: {
      topology: 'triangle-list',
    },
  });

  // For testing, set up an indirect draw buffer
  const vertices = new Float32Array([
    // x, y, z
    -0.5, -0.5, 0.0,
     0.5, -0.5, 0.0,
     0.0,  0.5, 0.0
  ]);


  const indirect_draw_data = new Uint32Array([
    3, 1, 0, 0 // vertexCount, instanceCount, firstVertex, firstInstance
  ]);

  const indirect_draw_buffer = createBuffer(
    device,
    'indirect draw buffer',
    4 * 4,
    GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    indirect_draw_data
  );

  const vertex_buffer = createBuffer(
    device,
    'vertex buffer',
    vertices.byteLength,
    GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    vertices
  );

  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    const pass = encoder.beginRenderPass({
      label: "render gaussian pass",
      colorAttachments: [
        {
          view: texture_view,
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    pass.setPipeline(render_pipeline);
    pass.setVertexBuffer(0, vertex_buffer);
    pass.drawIndirect(indirect_draw_buffer, 0);
    pass.end();
  }


  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      // sorter.sort(encoder);
      render(encoder, texture_view);
    },
    camera_buffer,
  };
}
