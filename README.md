# Project5-WebGPU-Gaussian-Splat-Viewer

University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4

* Matt Schwartz
* Tested on: Google Chrome on Windows 10 22H2, Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz, NVIDIA GeForce RTX 2060


### Live Demo

[![](img/thumb.png)](http://TODO.github.io/Project4-WebGPU-Forward-Plus-and-Clustered-Deferred)

### Demo Video/GIF

[![](img/video.mp4)](TODO)

# Background

Gaussian splatting is a somewhat new technique for continuous scene representations based on 3D volumetric data (captured from various camera viewpoints around a real scene). The key words here, "continuous scene representation," indicate that we can freely view the rendered scene from any angle, despite the fact that the camera data has been captured only from specific positions.

Other methods exist for achieving similar scene continuity such as Neural Radiance Fields (NeRFs), however, this class of methods rely on dense sampling of the scene along rays from the cameras as well as multilayer neural networks to train models to infer novel views of the scene. Such methods are slow to train and slow to render. Gaussian splatting, in contrast, can train its simplified model in a fraction of the time and render at 100+FPS.

This is the (highly simplified) overview of the process of capturing data to rendering a scene:
- Take images of a scene from various perspectives using lidar cameras and capture scene as a point cloud.
- Represent each point as a 3D gaussian witha mean and a variance in each direction (essentially "smearing" out the influence of the point over some space):

![Gaussians in different dimensionalities](/images/gaussians.png)

- Project the gaussians onto the 2D screen.
- Finally, render the gaussians, blending them together according to their alpha values.

In this project, we start with the the gaussian data and begin at step 3, projecting the gaussians into screen space. To keep this readme reader-friendly, however, I will mostly discuss the rendering step because the projection involves a fair amount of linear algebra. To learn more about that, read [this paper on gaussian splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

# Rendering Gaussians

To process each gaussian and compute its 2D conic projection before rendering them, we run a compute shader. We also take advantage of this preprocessing compute pass to store each gaussian by depth so we can later sort them. (This is necessary, since each gaussian is translucent, to blend them together in the right order). After sorting, we render each gaussian on individual quads centered on the gaussian. Each pixel in the quad is shaded by calculating its distance to the gaussian center and using our conic projection to adjust the opacity of that pixel appropriately.

Even though we're inferring view angles, a simple scene still contains a large amount of data to render. Some of the simplest test files linked above have 300MB of gaussian data. That's a lot of gaussians to process and render each frame - how do we operate on this data efficiently ?

1. Culling - the easiest optimization; don't draw the gaussians that aren't in view. We can do this in the preprocessing compute shader, before even doing a pass through the main render pipeline.
2. Indirect drawing - all of our data is already on the GPU, since we needed to preprocess and sort the gaussians. It would be wasteful to send it back to the CPU to issue draw commands, only to send the data right back to the GPU. Memory transfer is very expensive. Instead, we can leverage indirect draw calls, using the GPU buffer data to determine the contents of each draw.
3. Tiling (not implemented) - similar to modern tiled renderers, we could divide up the screen into tiles and tag each one with the gaussians overlapping them.

# Performance analysis

### Credits

- [Vite](https://vitejs.dev/)
- [loaders.gl](https://loaders.gl/)
- [dat.GUI](https://github.com/dataarts/dat.gui)
- [stats.js](https://github.com/mrdoob/stats.js)
- [wgpu-matrix](https://github.com/greggman/wgpu-matrix)
