# Project5-WebGPU-Gaussian-Splat-Viewer

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Deze Lyu
* Tested on: **Google Chrome Version 129.0.6668.101** on Windows 11, AMD Ryzen 5 5600U @ 2.30GHz 16GB, NVIDIA GeForce MX450 9142MB

### Live Demo

The live demo is deployed as a GitHub Pages website and can be accessed [here](https://dezelyu.github.io/CIS565-Project5/). Additionally, you need to prepare the PLY and CAM files, which can be downloaded from [this link](https://drive.google.com/drive/folders/1UbcFvkwZhcdnhAlpua7n-UUjxDaeiYJC?usp=sharing).

### Demo Animation

![](images/image0.gif)

### Deze Lyu

● **Compare your results from the point-cloud and Gaussian renderers. What are the differences?**

The Gaussian renderer is significantly slower than the point-cloud renderer for several reasons:

1. **Multiple Passes**: The Gaussian renderer requires multiple passes—the render pass reads from the preprocessed compute pass, adding to the overall processing time.

2. **Atomic Operations**: The preprocessing compute pass involves atomic operations, which require threads to wait when they attempt to operate on the same variable, leading to synchronization delays.

3. **Fragment Drawing**: Unlike point-cloud rendering, which draws points, the Gaussian renderer draws quads. This results in more fragments being processed, increasing the computational workload.

4. **Blending**: Blending is enabled in the Gaussian renderer, which significantly slows down performance. Each fragment must be blended with the fragments beneath it, adding to the rendering complexity and time.

● **For the Gaussian renderer, how does changing the workgroup size affect performance? Why do you think this is?**

On my machine, the most performant workgroup size is the default one; decreasing or increasing the workgroup size severely impacts performance. If the workgroup is too large, it leads to heavy memory consumption and resource contention. Conversely, if the workgroup size is too small, the compute shader needs to dispatch more workgroups. Due to the use of atomic operations, a smaller workgroup size can be more problematic, as it exacerbates synchronization issues and increases overhead.

● **Does view frustum culling provide a performance improvement? Why do you think this is?**

Frustum culling helps remove Gaussians that are outside the screen, improving performance. However, in relatively small scenes where all Gaussians are visible, such as the bicycle scene, performing frustum culling can be a waste of computation. In these cases, even atomic operations may become unnecessary and can hinder the program's performance. Conversely, in larger scenes where the camera is surrounded by a vast environment, frustum culling becomes essential to eliminate invisible Gaussians, thereby reducing the workload and enhancing performance.

● **Does the number of Gaussians affect performance? Why do you think this is?**

The total number of Gaussians does affect performance. First, loading these Gaussians consumes memory, and if the quantity is too large, memory access can negatively impact performance. In fact, my machine can only handle small and medium scene files, as larger scenes cause it to run out of memory. 

On the computing and rendering side, even with frustum culling, the preprocessing shader still has to process all the Gaussians to determine their visibility. Furthermore, the current implementation of frustum culling is solely position-based, meaning that small Gaussians near the edge of the screen are still considered, which can lead to unnecessary computations.

### Credits

- [Vite](https://vitejs.dev/): A fast build tool and development server for modern web projects.
- [loaders.gl](https://loaders.gl/): A suite of libraries for loading and processing geospatial data.
- [dat.GUI](https://github.com/dataarts/dat.gui): A lightweight GUI for changing variables in JavaScript.
- [stats.js](https://github.com/mrdoob/stats.js): A performance monitor for tracking frames per second and memory usage.
- [wgpu-matrix](https://github.com/greggman/wgpu-matrix): A WebGPU matrix math library for efficient computations.
