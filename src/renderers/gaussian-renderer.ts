import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {

    // declare the render settings buffer
    render_settings_buffer: GPUBuffer,
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

    // create the null buffer
    const null_buffer = createBuffer(
        device, 'null_buffer', 4,
        GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, nulling_data
    );
    
    // create the render settings buffer
    const render_settings_buffer = createBuffer(
        device, 'render_settings_buffer', 8,
        GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM, new Float32Array([
            1.0, pc.sh_deg,
        ])
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

    // create the camera bind group
    const camera_bind_group = device.createBindGroup({
        label: 'camera_bind_group',
        layout: preprocess_pipeline.getBindGroupLayout(0),
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: camera_buffer,
                },
            },
        ],
    });
    
    // create the gaussian bind group
    const gaussian_bind_group = device.createBindGroup({
        label: 'gaussian_bind_group',
        layout: preprocess_pipeline.getBindGroupLayout(1),
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: pc.gaussian_3d_buffer,
                },
            },
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

    // compute the splat data size
    var splat_data_size = 0;
    
    // increase the splat data size for the position and size data
    splat_data_size += 8;
    
    // increase the splat data size for the color data
    splat_data_size += 8;

    // create the splat data buffer
    const splat_data_buffer = createBuffer(
        device, 'splat_data_buffer', splat_data_size * pc.num_points,
        GPUBufferUsage.STORAGE, null
    );
    
    // create the compute pipeline bind group for all other resources
    const compute_pipeline_bind_group = device.createBindGroup({
        label: 'compute_pipeline_bind_group',
        layout: preprocess_pipeline.getBindGroupLayout(3),
        entries: [
            
            // declare a new entry for the splat data buffer
            {
                binding: 0,
                resource: {
                    buffer: splat_data_buffer,
                },
            },
            
            // declare a new entry for the render settings buffer
            {
                binding: 1,
                resource: {
                    buffer: render_settings_buffer,
                },
            },
            
            // declare a new entry for the color buffer
            {
                binding: 2,
                resource: {
                    buffer: pc.sh_buffer,
                },
            },
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
            0,
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
    
    // create the render pipeline bind group for all other resources
    const render_pipeline_bind_group = device.createBindGroup({
        label: 'render_pipeline_bind_group',
        layout: render_pipeline.getBindGroupLayout(0),
        entries: [
            
            // declare a new entry for the splat data buffer
            {
                binding: 0,
                resource: {
                    buffer: splat_data_buffer,
                },
            },
            
            // declare a new entry for the sort indices buffer
            {
                binding: 1,
                resource: {
                    buffer: sorter.ping_pong[0].sort_indices_buffer,
                },
            },
        ],
    });

  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  
    // create the compute function
    const compute = (encoder: GPUCommandEncoder) => {
        
        // begin a new compute pass
        const compute_pass = encoder.beginComputePass();
        
        // bind the preprocess pipeline
        compute_pass.setPipeline(preprocess_pipeline);
        
        // bind the camera bind group
        compute_pass.setBindGroup(0, camera_bind_group);
        
        // bind the gaussian bind group
        compute_pass.setBindGroup(1, gaussian_bind_group);
        
        // bind the sort bind group
        compute_pass.setBindGroup(2, sort_bind_group);
        
        // bind the bind group that contains all other resources
        compute_pass.setBindGroup(3, compute_pipeline_bind_group);
        
        // execute the preprocess shader
        compute_pass.dispatchWorkgroups(Math.ceil(pc.num_points / C.histogram_wg_size));
        
        // end the compute pass
        compute_pass.end();
    };

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
        
        // reset the sorting buffers
        encoder.copyBufferToBuffer(
            null_buffer, 0,
            sorter.sort_info_buffer, 0,
            4
        );
        encoder.copyBufferToBuffer(
            null_buffer, 0,
            sorter.sort_dispatch_indirect_buffer, 0,
            4
        );
        
        // perform preprocessing
        compute(encoder);
        
        // perform sorting
        sorter.sort(encoder);
        
        // update the indirect buffer
        encoder.copyBufferToBuffer(
            sorter.sort_info_buffer, 0,
            indirect_buffer, 4,
            4
        );
        
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
        
        // bind the bind group that contains all other resources
        render_pass.setBindGroup(0, render_pipeline_bind_group);
        
        // perform drawing
        render_pass.drawIndirect(indirect_buffer, 0);
        
        // end the render pass
        render_pass.end();
    },
    camera_buffer,
      
      // return the render settings buffer
      render_settings_buffer,
  };
}
