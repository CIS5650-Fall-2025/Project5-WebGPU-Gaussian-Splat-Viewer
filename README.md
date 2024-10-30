# Project5-WebGPU-Gaussian-Splat-Viewer

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 5**

* CARLOS LOPEZ GARCES
  * [LinkedIn](https://www.linkedin.com/in/clopezgarces/)
  * [Personal website](https://carlos-lopez-garces.github.io/)
* Tested on: Windows 11, 13th Gen Intel(R) Core(TM) i9-13900HX @ 2.20 GHz, RAM 32GB, NVIDIA GeForce RTX 4060, personal laptop.


### [Live Demo](carlos-lopez-garces.github.io/Penn-CIS-5650-Project5-WebGPU-Gaussian-Splat-Viewer)

[![](images/bicycle_1.png)](carlos-lopez-garces.github.io/Penn-CIS-5650-Project5-WebGPU-Gaussian-Splat-Viewer)

### Demo Video/GIF

| Bonsai | Bicycle |
|----------|----------|
| ![](images/bonsai_1.gif)   | ![](images/bicycle_1.gif)   |
| **Truck** | **Train** |
| ![](images/truck_1.gif)   | ![](images/train_1.gif)   |


| Bonsai | Bicycle |
|----------|----------|
| ![](images/bonsai_1.png)   | ![](images/bicycle_1.png)   |
| **Truck** | **Train** |
| ![](images/truck_1.png)   | ![](images/train_1.png)   |

### WebGPU Gaussian Splat and Point Cloud Viewer

A point cloud is the main input to the Gaussian splatting rendering process.

| Bonsai | Bicycle |
|----------|----------|
| ![](images/pc_bonsai_1.gif)   | ![](images/pc_bicycle_1.gif)   |
| **Truck** | **Train** |
| ![](images/pc_truck_1.gif)   | ![](images/pc_train_1.gif)   |

- Compare your results from point-cloud and gaussian renderer, what are the differences?
- For gaussian renderer, how does changing the workgroup-size affect performance? Why do you think this is?
- Does view-frustum culling give performance improvement? Why do you think this is? 
- Does number of guassians affect performance?  Why do you think this is? 

### Credits

- [Vite](https://vitejs.dev/)
- [loaders.gl](https://loaders.gl/)
- [dat.GUI](https://github.com/dataarts/dat.gui)
- [stats.js](https://github.com/mrdoob/stats.js)
- [wgpu-matrix](https://github.com/greggman/wgpu-matrix)
