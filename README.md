# Project5-WebGPU-Gaussian-Splat-Viewer

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 5**

* Maya Diaz Huizar
* Tested on: Google Chrome 132.0 on Windows 10, AMD Ryzen 9 5900X @ 3.7GHz 32GB RAM, Nvidia RTX 3080 10GB 

## Live Demo (requires Chrome and a WebGPU compatible GPU)
[Demo Link](https://aorus1.github.io/Project5-WebGPU-Gaussian-Splat-Viewer/)
[Sample Files](https://drive.google.com/drive/folders/1rwtArEbj7GfjMeK6mD3a4QN6ZFLKQoBB?usp=sharing)

## Demo Video

[![Demo Video]()](images/demo.mp4)

## Project Overview

The WebGPU Gaussian Splat Viewer is a 3D renderer for Gaussian splats, designed to display and visualize point cloud data using a Gaussian splatting technique. Gaussian splats allow for smooth, realistic rendering of point clouds with transparency effects. This viewer is built in WebGPU, featuring a Gaussian renderer that includes preprocessing and rendering pipelines, depth sorting, and spherical harmonics-based shading.

## Features

### Gaussian Preprocessing Pipeline
The preprocessing pipeline handles 3D Gaussian data before rendering:
- **View-Frustum Culling**: Removes non-visible Gaussians outside the camera’s view.
- **3D to 2D Transformation**: Transforms Gaussian points into 2D screen space for rendering.
- **Covariance Calculation**: Computes a 2D conic for each Gaussian based on a covariance matrix derived from 3D rotation and scaling data.
- **Opacity Scaling**: Applies a sigmoid function to Gaussian opacities for realistic transparency.
- **Depth Sorting**: Uses a GPU-based radix sort to sort Gaussians by depth for accurate transparency effects.

## Performance Analysis
- **Point Cloud vs. Gaussian Renderer**: Gaussian splatting improves rendering quality for point clouds, with spherical harmonics and Gaussian opacity scaling providing smoother, more realistic visuals. However, the Gaussian renderer is more computationally intensive than direct point cloud rendering due to shading and transparency effects. Performance is obviously and noticeably worse, but this is understandable.
  
- **Effect of Workgroup Size on Performance**: Workgroup size significantly impacts performance in the Gaussian renderer. I found the default parameter given to be the most performant, which makes sense as its likely a nice middle ground. 

- **View-Frustum Culling**: Culling offers a substantial performance boost by eliminating off-screen Gaussians from rendering, reducing the number of draw calls and unnecessary GPU computations. The speedup is substantial because a significant number of threads can be abandoned, leading to faster draws. 

- **Gaussian Count and Performance**: As the number of Gaussians increases, rendering performance decreases due to increased depth sorting and fragment processing requirements. Testing with varying Gaussian counts reveals the expected worse performance with increased Gaussian counts. 

## Credits
- [Vite](https://vitejs.dev/)
- [tweakpane](https://cocopon.github.io/tweakpane/)
- [wgpu-matrix](https://github.com/greggman/wgpu-matrix)
- [WebGPU Developer Tools](https://chrome.google.com/webstore/detail/webgpu-developer-tools/)
- Special thanks to Shrek Shao from Google WebGPU and [Differential Gaussian Renderer](https://github.com/graphdeco-inria/diff-gaussian-rasterization) for inspiration and resources.
