# WebGPU-Gaussian-Splat-Viewer

University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 5

* Matt Schwartz
* Tested on: Google Chrome on Windows 10 22H2, Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz, NVIDIA GeForce RTX 2060


[![](images/GaussianSplatting.gif)](TODO)


### Live Demo

[Click for link to live demo webpage](https://mzschwartz5.github.io/WebGPU-Gaussian-Splat-Viewer/)

(Note: you will need to download a `.ply` file from [this drive](https://drive.google.com/drive/u/2/folders/1UbcFvkwZhcdnhAlpua7n-UUjxDaeiYJC), along with its associated `cameras.json` file. Then, go to the above site and upload those files (may take half a minute or so). Finally, select "gaussian" from the drop down).

# Background

Gaussian splatting is a somewhat new technique for continuous scene representations based on 3D volumetric data (captured from various camera viewpoints around a real scene). The key words here, "continuous scene representation," indicate that we can freely view the rendered scene from any angle, despite the fact that the camera data has been captured only from specific positions.

Other methods exist for achieving similar scene continuity such as Neural Radiance Fields (NeRFs), however, that class of methods relies on dense sampling of the scene along rays from the cameras as well as multilayer neural networks to train models to infer novel views of the scene. Such methods are slow to train and slow to render. Gaussian splatting, in contrast, can train its simplified model in a fraction of the time and render at 100+FPS.

This is the (highly simplified) overview of the process of capturing data to rendering a scene:
- Take images of a scene from various perspectives using lidar cameras and capture scene as a point cloud.
- Represent each point as a 3D gaussian witha mean and a variance in each direction (essentially "smearing" out the influence of the point over some space):

![Gaussians in different dimensionalities](/images/gaussians.png)

- Project the gaussians onto the 2D screen.
- Finally, render the gaussians, blending them together according to their alpha values.

In this project, we start with the the gaussian data and begin at step 3, projecting the gaussians into screen space. To keep this readme reader-friendly, however, I will mostly discuss the rendering step because the projection involves a fair amount of linear algebra. To learn more about that, read [this paper on gaussian splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).

Some scenes rendered via gaussian splatting:

![Bicycle gaussian splatting](/images/bicycle.PNG)

![Bonsai gaussian splatting](/images/gaussiansplattingbonsai.PNG)

# Rendering Gaussians

To process each gaussian and compute its 2D conic projection before rendering them, we run a compute shader. We also take advantage of this preprocessing compute pass to store each gaussian by depth so we can later sort them. (This is necessary, since each gaussian is translucent, to blend them together in the right order). After sorting, we render each gaussian on individual quads centered on the gaussian. Each pixel in the quad is shaded by calculating its distance to the gaussian center and using our conic projection to adjust the opacity of that pixel appropriately.

Even though we're inferring view angles, a simple scene still contains a large amount of data to render. Some of the simplest test files linked above have 300MB of gaussian data. That's a lot of gaussians to process and render each frame - how do we operate on this data efficiently ?

1. Culling - the easiest optimization; don't draw the gaussians that aren't in view. We can do this in the preprocessing compute shader, before even doing a pass through the main render pipeline.
2. Indirect drawing - all of our data is already on the GPU, since we needed to preprocess and sort the gaussians. It would be wasteful to send it back to the CPU to issue draw commands, only to send the data right back to the GPU. Memory transfer is very expensive. Instead, we can leverage indirect draw calls, using the GPU buffer data to determine the contents of each draw.
3. Tiling (not implemented) - similar to modern tiled renderers, we could divide up the screen into tiles and tag each one with the gaussians overlapping them.

# Performance analysis

To test, I'm simply taking the average frame render time over a 100-frame rolling window.

## Basic point cloud vs. Gaussian splats

Same camera perspective, `bicycle_30000.cleaned.ply`:

Basic point cloud: 4.16 milliseconds per frame

Gaussian splats: 12.03 milliseconds per frame

It makes sense that the gaussian splats would render more slowly than a basic point cloud for several reasons:
1. The gaussians are rendered on full quads, whereas the points are only points.
2. The gaussians require alpha blending.
3. The gaussians require preprocessing and sorting, both of which are computation- *and* memory-intensive.

## Workgroup size impact

Same camera perspective, `bicycle_30000.cleaned.ply`:

Workgroup size : render time (milliseconds per frame)
```
256 : 12.03 
128: 56.3
64: 76.9
```

Not totally surprising, the more threads in a workgroup, the faster the work gets done. However, compared to other projects I've done on the GPU, this is by far the biggest increase in speed I've seen due to increasing workgroup size. It may imply that the workload is more compute-heavy than other projects I've done, and less bound by memory latency.

## View frustum culling

This is hard to quantify. Qualitatively, I saw the framerate increase dramatically as I zoomed in. However, I also saw it increase dramatically as I zoomed out. For the latter, that tells me that the fragment shader is incurring a large cost; as we zoom out and the gaussians occupy a smaller portion of the screen, fewer pixels need fragment shading, so the framerate increases. For the former, as we zoom in, more gaussians can be culled and thus our preprocess, sort, and render steps all have less work to do.

I believe that, because I expanded the culling distance to 1.2x the view frustum, the impact of the culling is significantly dampened. This was done to avoid clipping edge-gaussians, but at the cost of some performance.

## Gaussian count

Again, hard to quantify because I can't say how many gaussians are on screen, exactly. That said, qualitatively, as the number of gaussians on screen increases, the performance drops. This, again, makes sense, because we have more gaussians to process, sort, render, and blend.

# Bloopers

Before I remembered I had to turn on alpha blending! Whoops!

![Blooper image](/images/blooper.webp)


### Credits

- [Vite](https://vitejs.dev/)
- [loaders.gl](https://loaders.gl/)
- [dat.GUI](https://github.com/dataarts/dat.gui)
- [stats.js](https://github.com/mrdoob/stats.js)
- [wgpu-matrix](https://github.com/greggman/wgpu-matrix)
