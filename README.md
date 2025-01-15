WebGPU Gaussian Splat Viewer
============================

**University of Pennsylvania, CIS 5650: GPU Programming and Architecture: Project 5**

* Dominik Kau ([LinkedIn](https://www.linkedin.com/in/dominikkau/))
* Tested on: **Google Chrome 132.0**, macOS Sequoia 15.1.1, Apple M3 Pro

## Live Demo

[Click here to check out my implementation!](https://domino0o.github.io/Project5-WebGPU-Gaussian-Splat-Viewer/)
You'll need some trained gaussian splat input data.

## Overview

Gaussian splatting is a technique for reconstructing a 3D scene from images.
The method consists of a training period during which 3-dimensional Gaussian distributions are placed around the scene and optimized to best capture the given image data set.
In this phase the "nice" mathematical properties of a Gaussian distribution can be used to calculate gradients with respect to the target variables such as position, color and size of the Gaussian.
Thus, gradient based optimization techniques can be used.

Afterwards the camera can be moved freely throughout the reconstructed scene.
This allows the synthesis of images from new perspectives or videos that can be recorded in real-time with the methods presented in [this paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).
This project implements a renderer to display a scene consisting of 3D Gaussians after having been trained.

## Features

### Point Cloud Renderer

![Image of a scene containing a bench and bicycle visualized only with dots.](images/points.png)

The point cloud renderer is a simple visualization of the Gaussians that are present in the dataset.
Each visible Gaussian is displayed as a point placed in its center.

### Gaussian Renderer

![Reconstructed color image of a scene containing a bench and bicycle.](images/gaussian.png)

The main feature of this project.
The Gaussian renderer extracts the information from the given dataset and - next to position - reconstructs color (through spherical harmonics) and size of the Gaussians.
The reconstruction of information takes place in a preprocessing compute shader before the render pipeline is carried out.
A simple view-frustum culling is used to reduce the computations in the render pass.
The final rendering pipeline employs alpha blending to composite the Gaussians into an image.
The alpha blending requires the Gaussians to be rendered from furthest back to the closest.
For this, another compute shader is used that sorts the visible Gaussians based on their depth.

### Half Precision Floating Point Optimization

This optimization feature changes the preprocessing compute shader to operate on half precision floating point numbers.
This allows for a smaller memory footprint - instead of 32 bits, only 16 bits are needed.
Moreover, modern GPUs can run operations of half precision floats significantly faster.
Not all computations are carried out using 16-bit floats, as some mathematical operations need the higher accuracy to ensure an accurate result without artifacts.

As a result there are only very slight visual differences between the 16-bit and 32-bit version.

## Performance Analysis

### Comparison Between Point Cloud and Gaussian Renderer

Obviously, the performance of the point cloud renderer is significantly better than the Gaussian renderer.
The point cloud renderer doesn't have a preprocessing or sorting compute shader that needs to run, so it performs drastically fewer computations.

### Influence of Workgroup Size

Decreasing the workgroup size leads to a small performance hit.
As this project does not rely on shared memory, this is probably due to how the GPU dispatching is handled.
Smaller workgroups will lead to more work on the dispatching side.

### Influence of View-Frustum Culling

The view frustum culling has a significant effect on the performance.
This is especially true for indoor scenes, because there will be many points present in the dataset that don't have to be rendered to the camera.
Because the view-frustum culling is implemented at the beginning of the preprocessing compute shader, a lot of computations can be avoided by returning early.
However, this is also dependent on the degree of thread divergence, because the effect of an early return will only have a big impact on performance if multiple all "connected" threads (in CUDA: a warp) quit early.

### Influence of Number of Gaussians

The number of Gaussians has a significant impact on performance.
Firstly, all buffers are sized according to the total number of points - leading to memory bottlenecks.
Secondly, the more Gaussians are visible at the same time, the more computations the compute shaders and the rendering shaders have to carry out.
However, the second point is scene dependent because of view-frustum culling.
A scene with few total but spatially concentrated Gaussians will run at a similar speed as a scene with more but evenly distributed Gaussians, that are not all visible at the same time.
That assumes that memory does not pose an issue.

### Influence of Using Half Precision Floats

The usage of half precision floats increases performance, but not as much as expected.
This might be because I'm still using 32-bit computations in operations that I found critical for accurate visual output.

### Credits

- [Vite](https://vitejs.dev/)
- [tweakpane](https://tweakpane.github.io/docs//v3/monitor-bindings/)
- [stats.js](https://github.com/mrdoob/stats.js)
- [wgpu-matrix](https://github.com/greggman/wgpu-matrix)
- Special Thanks to: Shrek Shao (Google WebGPU team) & [Differential Guassian Renderer](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
