# Project5-WebGPU-Gaussian-Splat-Viewer

**University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Project 3**

* Shreyas Singh
    * [LinkedIn](https://linkedin.com/in/shreyassinghiitr)
* Tested on: Apple MacBook Pro, Apple M2 Pro @ 3.49 GHz, 19-core GPU


### Live Demo

[![](img/thumb.png)](http://TODO.github.io/Project4-WebGPU-Forward-Plus-and-Clustered-Deferred)

### Demo Video/GIF

[![](img/video.mp4)](TODO)

### Project Overview

#### Gaussian Splatting

Gaussian splatting is a way to draw 3D scenes by treating each point as a little “splat” instead of a triangle. Each splat is a 3D Gaussian defined by its center, shape, and color (often encoded with spherical harmonics). To render, we transform these Gaussians into screen space, drop the ones that lie outside the view, sort the remaining Gaussians back-to-front, and draw them as small quads whose opacity falls off smoothly from the center.

#### Project Overview
This WebGPU project builds a real-time viewer for a pre-trained Gaussian scene. First, a compute pass culls off-screen Gaussians, computes each splat’s 2D position, size, and color, and builds an indirect draw command. Then a single indirect draw call runs a vertex shader to place each quad and a fragment shader to shade it using the Gaussian equation. The GUI sliders let us tweak the global scale and color detail on the fly, so we can interactively explore the reconstructed scene.



### Credits

- [Vite](https://vitejs.dev/)
- [tweakpane](https://tweakpane.github.io/docs//v3/monitor-bindings/)
- [stats.js](https://github.com/mrdoob/stats.js)
- [wgpu-matrix](https://github.com/greggman/wgpu-matrix)
- Special Thanks to: Shrek Shao (Google WebGPU team) & [Differential Guassian Renderer](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
