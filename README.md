# Project5-WebGPU-Gaussian-Splat-Viewer

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 5**
* Jiahang Mao
  * [LinkedIn](https://www.linkedin.com/in/jay-jiahang-m-b05608192/)
* Tested on: Windows 11, i5-13600kf @ 5.0GHz 64GB, RTX 4090 24GB, Personal Computer


### Live Demo

https://hazard-10.github.io/Project5-WebGPU-Gaussian-Splat-Viewer/

### Demo Video/GIF

https://drive.google.com/file/d/1wptImlSMpFDsPeE5Ss_jZy5fSTjJbD1n/view?usp=sharing

### Intro

This project implements an online viewer of Gaussian Splats file with webgpu as backend. User may easily open and preview .ply gs files in their browsers.

![](images/bike.jpg)

### Performance analysis
* Compare your results from point-cloud and gaussian renderer, what are the differences?

  Visual-wise, gaussian renderer show much better color and lighting. Performances-wise, while point-cloud can run at steady 16ms render time ( 60 fps), gaussian renderer is slightly slower at about 21ms render time per frame.

* For gaussian renderer, how does changing the workgroup-size affect performance? Why do you think this is?

  Changin the workgroup-size from 256 to 64 and 512, I see no discernable differences in performances. All three mantained around 20 to 21ms render time.

* Does view-frustum culling give performance improvement? Why do you think this is?

  Between view-frustum culling on and off, I see no discernable difference in performance. I think it is because with culling on, while we have less spat to check against, we have more fragments on the screen that need to be alpha blend, since each individual splat takes more space on the screen. 

* Does number of guassians affect performance? Why do you think this is?

  I think the answer depends. When reducing number of splats by setting a hard limit of 3000 splats ( 1/ 10 of the scene ) in my preprocess shader, i see no difference in render time. I think that is because GPU still need to go through serialized execution of preprocess, sorting and rendering which is the bottleneck. But when increasing the number splats from bicycle to kitchen, the render time increased from 20 to 41 ms per second. Clearly more cycles are needed to cover the full range of splats.
  
### Credits

- [Vite](https://vitejs.dev/)
- [tweakpane](https://tweakpane.github.io/docs//v3/monitor-bindings/)
- [stats.js](https://github.com/mrdoob/stats.js)
- [wgpu-matrix](https://github.com/greggman/wgpu-matrix)
- Special Thanks to: Shrek Shao (Google WebGPU team) & [Differential Guassian Renderer](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
