import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

const c_size_splat = 4 * 2 * 3;

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
  const reset_buffer = createBuffer(
		device, 'reset buffer', 4, GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, nulling_data
	);

	const splat_buffer = createBuffer(
		device, 'splat buffer', pc.num_points*c_size_splat, GPUBufferUsage.STORAGE
	);

  const indirect_render_buffer = createBuffer(
		device, 'indirect buffer', 16, GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
		new Uint32Array([6, 0, 0, 0])
	);

	const settings_buffer = createBuffer(
		device, 'settings buffer', 8, GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM, 
		new Float32Array([1.0, pc.sh_deg])
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

  const preprocess_bind_group = device.createBindGroup({
    label: 'preprocess bind group',
    layout: preprocess_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },
      { binding: 1, resource: { buffer: settings_buffer } },
      { binding: 2, resource: { buffer: pc.sh_buffer } },
      { binding: 3, resource: { buffer: pc.gaussian_3d_buffer } },
      { binding: 4, resource: { buffer: splat_buffer } }
    ],
  });

  const sort_bind_group = device.createBindGroup({
    label: 'sort bind group',
    layout: preprocess_pipeline.getBindGroupLayout(1),
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
    encoder.copyBufferToBuffer(reset_buffer, 0, sorter.sort_info_buffer, 0, 4);
		encoder.copyBufferToBuffer(reset_buffer, 0, sorter.sort_dispatch_indirect_buffer, 0, 4);

		const pass = encoder.beginComputePass({ label: 'preprocess pass' });
		pass.setPipeline(preprocess_pipeline);
		pass.setBindGroup(0, preprocess_bind_group);
		pass.setBindGroup(1, sort_bind_group);
		pass.dispatchWorkgroups(Math.ceil(pc.num_points / C.histogram_wg_size));
		pass.end();
  };

  const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, indirect_render_buffer, 4, 4);

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: texture_view,
        loadOp: 'clear',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        storeOp: 'store'
      }],
    });

    pass.setPipeline(render_pipeline);
    pass.setBindGroup(0, render_bind_group);
    pass.drawIndirect(indirect_render_buffer, 0);
    pass.end();
  };

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      preprocess(encoder);
      sorter.sort(encoder);
      render(encoder, texture_view);
    },
    camera_buffer,
  };
}
