# WebGPU Gaussian Splat Viewer

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 5**

* Mufeng Xu
* Tested on: **Google Chrome 129.0** on
  Windows 11, i9-13900H @ 2.6GHz 32GB, RTX 4080 Laptop 12282MB (Personal Computer)

## Live Demo

[Live Demo](https://solemnwind.github.io/Project5-WebGPU-Gaussian-Splat-Viewer/) (Open with Chrome)

## Demo Video/GIF

![](images/bonsai.gif)

## Project Overview

Gaussian Splatting is a volume rendering technique that deals with the direct rendering of volume data without converting the data into surface or line primitives. This method uses 3D gaussian distributions to represent objects, with spherical harmonics, the gaussian splats can represent sophisticated colors and details. This project specifically implements a viewer that renders a gaussian splat scene.

**Implemented Functions**

* Add MVP calculation to Point Cloud rendering
* View frustum culling to remove non-visible splats
* Use spherical harmonics to evaluate colors of splats
* Gaussian Splatter renderer

## Performance Analysis

### Compare your results from point-cloud and gaussian renderer, what are the differences?

The point-cloud renderer shows sparse points, these points typically don't have volumes and can't block each other,
and they don't have transparency and color (but color is possible, though not detailed).
While the gaussian renderer renders globs at the positions of the points, the scene is smoother and much more realistic.
Because the use of spherical harmonics for representing the colors, gaussian splatting renders detailed colors with a rather low cost.

### For gaussian renderer, how does changing the workgroup-size affect performance? Why do you think this is?

Changing the workgroup size larger, generally makes the GPU more utilized, therefore improve the performance. However, if the workgroup is too large, first the increased data contention will slow down the performance, second, for some scenes, an increased workgroup causes insufficient buffer space, and the renderer does not work in that case. While decreasing the workgroup size will lead to less parallel computing and lower the FPS.

### Does view-frustum culling give performance improvement? Why do you think this is?

In the cases where the camera is *inside* the scene, such as the room, view frustum culling could lower the GPU usage, and potentially improve the FPS if the scene is large. And this is also what I observed: even for a simple scene, why zoomed out and include the whole scene in the screen, the GPU usage rises immediately, and for a larger scene, it causes lower FPS. However, view-frustum culling increases overhead, potentially worsen the improvement in simple scenes or the case that most of the gaussians are in the view-frustum.

### Does number of gaussians affect performance? Why do you think this is?

When the number is not too large, because of the parallelism, it doesn't impact the performance. In my case, before the buffer space ran out, the FPS was always steady at ~120. But the increased number of gaussians makes culling, sorting etc. more expensive, and therefore impact the performance.

## Credits

* [Vite](https://vitejs.dev/)
* [tweakpane](https://tweakpane.github.io/docs//v3/monitor-bindings/)
* [stats.js](https://github.com/mrdoob/stats.js)
* [wgpu-matrix](https://github.com/greggman/wgpu-matrix)
* Special Thanks to: Shrek Shao (Google WebGPU team) & [Differential Guassian Renderer](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
