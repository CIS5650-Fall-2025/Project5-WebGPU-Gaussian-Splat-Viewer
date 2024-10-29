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
  const splat_buffer_size = 28;
  const nulling_data = new Uint32Array([0]);
  let splat_buffer = createBuffer(device,
                        'splat buffer',
                        pc.num_points*splat_buffer_size,
                        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST

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
  
  const render_shader = device.createShaderModule({code:renderWGSL });

  const camera_bind_group_layout = device.createBindGroupLayout({
    label:'camera layout',
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: {
          type: 'uniform'
        }
      }
    ]
  });

  
  const gaussianBindGroupLayout = device.createBindGroupLayout({
    label: 'gaussian bind group layout',
    entries: [
        {
            // For gaussian_3d_buffer
            binding: 0,
            visibility: GPUShaderStage.VERTEX,
            buffer: {
                type: 'read-only-storage'
            }
        },
        {
            // For splat_buffer
            binding: 1,
            visibility: GPUShaderStage.VERTEX,
            buffer: {
                type: 'read-only-storage'
            }
        },
        {
            // For sh_buffer
            binding: 2,
            visibility: GPUShaderStage.VERTEX,
            buffer: {
                type: 'read-only-storage'
            }
        }
    ]
  });

  const pipelineLayout = device.createPipelineLayout({
    label: 'render pipeline layout',
    bindGroupLayouts: [
        camera_bind_group_layout, // group 0
        gaussianBindGroupLayout   // group 1
    ]
});

  const render_pipeline = device.createRenderPipeline({
    label: 'render',
    layout: pipelineLayout,
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
      topology: 'point-list',
    },
  });

  const camera_bind_group = device.createBindGroup({
    label: 'point cloud camera',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [{binding: 0, resource: { buffer: camera_buffer }}],
  });


  const gaussian_bind_group = device.createBindGroup({
    label: 'point cloud gaussians',
    layout: gaussianBindGroupLayout,
    entries: [
      {binding: 0, resource: { buffer: pc.gaussian_3d_buffer }},
      {binding: 1, resource: {buffer: splat_buffer}},
      {binding:2,resource:{buffer:pc.sh_buffer}}
    ],
  });

  const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    const pass = encoder.beginRenderPass({
      label: 'point cloud render',
      colorAttachments: [
        {
          view: texture_view,
          loadOp: 'clear',
          storeOp: 'store',
        }
      ],
    });
    pass.setPipeline(render_pipeline);
    pass.setBindGroup(0, camera_bind_group);
    pass.setBindGroup(1, gaussian_bind_group);

    pass.draw(pc.num_points);
    pass.end();
  };
  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      sorter.sort(encoder);
      render(encoder,texture_view);
    },
    camera_buffer,
  };
}
