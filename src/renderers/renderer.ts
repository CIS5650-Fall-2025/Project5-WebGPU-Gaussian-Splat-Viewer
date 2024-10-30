import { load } from '../utils/load';
import { Pane } from 'tweakpane';
import * as TweakpaneFileImportPlugin from 'tweakpane-plugin-file-import';
import { default as get_renderer_gaussian, GaussianRenderer } from './gaussian-renderer';
import { default as get_renderer_pointcloud } from './point-cloud-renderer';
import { Camera, load_camera_presets } from '../camera/camera';
import { CameraControl } from '../camera/camera-control';
import { time, timeReturn } from '../utils/simple-console';

export interface Renderer {
  frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => void,
  camera_buffer: GPUBuffer,
}

export default async function init(
  canvas: HTMLCanvasElement,
  context: GPUCanvasContext,
  device: GPUDevice
) {
  // Track loading status of files
  let ply_file_loaded = false; 
  let cam_file_loaded = false; 

  // Renderer objects for point cloud and Gaussian rendering
  let renderers: { pointcloud?: Renderer, gaussian?: Renderer } = {};
  let gaussian_renderer: GaussianRenderer | undefined; 
  let pointcloud_renderer: Renderer | undefined; 
  let renderer: Renderer | undefined; 
  let cameras;

  // Initialize camera and controls
  const camera = new Camera(canvas, device);
  const control = new CameraControl(camera);

  // Set up canvas resize handling
  const observer = new ResizeObserver(() => {
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
    camera.on_update_canvas();
  });
  observer.observe(canvas);

  // Configure canvas context for rendering
  const presentation_format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format: presentation_format,
    alphaMode: 'opaque',
  });

  // ===============================================
  //    Tweakpane for Interactive Parameter Control
  // ===============================================

  const params = {
    fps: 0.0,
    gaussian_multiplier: 1,
    renderer: 'pointcloud',
    ply_file: '',
    cam_file: '',
  };

  const pane = new Pane({
    title: 'Config',
    expanded: true,
  });
  pane.registerPlugin(TweakpaneFileImportPlugin);

  // Monitor frame rate
  pane.addMonitor(params, 'fps', { readonly: true });

  // Renderer selection (pointcloud or gaussian)
  pane.addInput(params, 'renderer', {
    options: { pointcloud: 'pointcloud', gaussian: 'gaussian' }
  }).on('change', (e) => {
    renderer = renderers[e.value];
  });

  // Load PLY file for point cloud
  pane.addInput(params, 'ply_file', {
    view: 'file-input',
    lineCount: 3,
    filetypes: ['.ply'],
    invalidFiletypeMessage: "We can't accept those filetypes!"
  }).on('change', async (file) => {
    const uploadedFile = file.value as unknown as File;
    if (uploadedFile) {
      const pc = await load(uploadedFile, device);
      pointcloud_renderer = get_renderer_pointcloud(pc, device, presentation_format, camera.uniform_buffer);
      gaussian_renderer = get_renderer_gaussian(pc, device, presentation_format, camera.uniform_buffer);
      renderers = { pointcloud: pointcloud_renderer, gaussian: gaussian_renderer };
      renderer = renderers[params.renderer];
      ply_file_loaded = true;
    } else {
      ply_file_loaded = false;
    }
  });

  // Load camera presets from JSON file
  pane.addInput(params, 'cam_file', {
    view: 'file-input',
    lineCount: 3,
    filetypes: ['.json'],
    invalidFiletypeMessage: "We can't accept those filetypes!"
  }).on('change', async (file) => {
    const uploadedFile = file.value as unknown as File;
    if (uploadedFile) {
      cameras = await load_camera_presets(uploadedFile);
      camera.set_preset(cameras[0]);
      cam_file_loaded = true;
    } else {
      cam_file_loaded = false;
    }
  });

  // Control Gaussian multiplier
  pane.addInput(params, 'gaussian_multiplier', { min: 0, max: 1.5 }).on('change', (e) => {
    if (gaussian_renderer) {
      device.queue.writeBuffer(
        gaussian_renderer.render_settings_buffer,
        0,
        new Float32Array([e.value])
      );
    }
  });

  // ===============================================
  //    Camera Preset Shortcuts (0-9 Keys)
  // ===============================================

  document.addEventListener('keydown', (event) => {
    const key = parseInt(event.key);
    if (!isNaN(key) && key >= 0 && key <= 9) {
      console.log(`Set to camera preset ${key}`);
      camera.set_preset(cameras[key]);
    }
  });

  // ===============================================
  //    Render Loop with Frame Timing
  // ===============================================

  function frame() {
    if (ply_file_loaded && cam_file_loaded) {
      // Update FPS display
      params.fps = 1.0 / timeReturn() * 1000.0;
      time();

      // Create command encoder and render the current frame
      const encoder = device.createCommandEncoder();
      const texture_view = context.getCurrentTexture().createView();
      renderer.frame(encoder, texture_view);
      device.queue.submit([encoder.finish()]);
    }
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}
