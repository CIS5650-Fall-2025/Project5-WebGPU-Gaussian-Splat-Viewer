# Project5-WebGPU-Gaussian-Splat-Viewer

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Joshua Smith
  * [LinkedIn](https://www.linkedin.com/in/joshua-smith-32b165158/)
* Tested on: Ubuntu 20.04, Ryzen 9 3900x @ 4.6GHz, 24GB RTX 4090 (Personal)
### Live Demo (Click image for online demo)
You need to download scene/camera files available [here](https://drive.google.com/drive/folders/1UbcFvkwZhcdnhAlpua7n-UUjxDaeiYJC?usp=sharing)

[![](images/bonsai_link_img.png)](https://joshmsmith44.github.io/Project5-WebGPU-Gaussian-Splat-Viewer/)

### Demo Video

![](images/gaussian_splatting_demo.gif)

### README

Features Implemented:
* Point Cloud Renderer: Render a set of points in the scene.
![](images/point_set_img.png)
* Gaussian Splatting Renderer: Given a large set of 3D gaussians, with spherical harmonic coloring, opacity, position and covariance, render in the scene. (see above examples)

How Gaussian Splatting Rendering is achieved:
* First, calculate all 3D gaussians 3D covariance and positions in the camera frame
* Cull (remove) gaussians which do not appear in the frame.
* Project, using the first-order Jacobian approximation, the 3D gaussians into 2D camera space.
* Calculate the gaussian color with spherical harmonic equations and provided gaussian coefficients. 
*  Render pixel color within gaussian tiles (which enclose >97% of the pdf) according to the 2D gaussian pdf taken from the previous projection.

### Performance Analysis


### Credits

- [Vite](https://vitejs.dev/)
- [tweakpane](https://tweakpane.github.io/docs//v3/monitor-bindings/)
- [stats.js](https://github.com/mrdoob/stats.js)
- [wgpu-matrix](https://github.com/greggman/wgpu-matrix)
- Special Thanks to: Shrek Shao (Google WebGPU team) & [Differential Guassian Renderer](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
