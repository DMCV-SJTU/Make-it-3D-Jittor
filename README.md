# Make-It-3D Jittor Implementation:

We provide [Jittor](https://github.com/Jittor/jittor) implementations for our paper "Make-It-3D: High-Fidelity 3D Creation from A Single Image with Diffusion Prior".


<!-- ![Teaser](teaser.png) -->
<div class="half">
    <img src="demo/bunny-cake.png" width="128"><img src="demo/bunny-cake-rgb.gif" width="128"><img src="demo/bunny-cake-normal.gif" width="128"><img src="demo/castle.png" width="128"><img src="demo/castle-rgb.gif" width="128"><img src="demo/castle-normal.gif" width="128">
</div>
<div class="half">
    <img src="demo/house.png" width="128"><img src="demo/house-rgb.gif" width="128"><img src="demo/house-normal.gif" width="128"><img src="demo/jay.png" width="128"><img src="demo/jay-rgb.gif" width="128"><img src="demo/jay-normal.gif" width="128">
</div>

### [Project page](https://make-it-3d.github.io/) |   [Paper](https://arxiv.org/abs/2303.14184) 
<!-- <br> -->
[Junshu Tang](https://junshutang.github.io/), [Tengfei Wang](https://tengfei-wang.github.io/), [Bo Zhang](https://bo-zhang.me/), [Ting Zhang](https://www.microsoft.com/en-us/research/people/tinzhan/), [Ran Yi](https://yiranran.github.io/), [Lizhuang Ma](https://dmcv.sjtu.edu.cn/), and [Dong Chen](https://www.microsoft.com/en-us/research/people/doch/).
<!-- <br> -->


## Abstract
>In this work, we investigate the problem of creating high-fidelity 3D content from only a single image. This is inherently challenging: it essentially involves estimating the underlying 3D geometry while simultaneously hallucinating unseen textures. To address this challenge, we leverage prior knowledge from a well-trained 2D diffusion model to act as 3D-aware supervision for 3D creation. Our approach, Make-It-3D, employs a two-stage optimization pipeline: the first stage optimizes a neural radiance field by incorporating constraints from the reference image at the frontal view and diffusion prior at novel views; the second stage transforms the coarse model into textured point clouds and further elevates the realism with diffusion prior while leveraging the high-quality textures from the reference image. Extensive experiments demonstrate that our method outperforms prior works by a large margin, resulting in faithful reconstructions and impressive visual quality. Our method presents the first attempt to achieve high-quality 3D creation from a single image for general objects and enables various applications such as text-to-3D creation and texture editing.





## Todo (Latest update: 2024/06/07)
- [x] **Release coarse stage training code**
- [ ] **Release all training code (coarse + [refine stage](#refine-stage))**

## Demo of 360° geometry
<div class="half">
    <img src="demo/teddy.png" width="128"><img src="demo/teddy-rgb.gif" width="128"><img src="demo/teddy-normal.gif" width="128"><img src="demo/teddy-2.png" width="128"><img src="demo/teddy-2-rgb.gif" width="128"><img src="demo/teddy-2-normal.gif" width="128">
</div>

## SAM + Make-It-3D
<div class="half">
    <img src="demo/corgi-demo.png" height="170"><img src="demo/corgi.png" width="170"><img src="demo/corgi-rgb.gif" width="170"><img src="demo/corgi-normal.gif" width="170">
</div>


## Installation
Install with pip:
```
    pip install git+https://github.com/JittorRepos/jittor
    pip install git+https://github.com/JittorRepos/jtorch
    pip install git+https://github.com/JittorRepos/diffusers_jittor
    pip install git+https://github.com/JittorRepos/transformers_jittor
    pip install git+https://github.com/openai/CLIP.git
    pip install git+https://github.com/huggingface/diffusers.git
    pip install git+https://github.com/huggingface/huggingface_hub.git
```
Other dependencies:
```
    pip install -r requirements.txt 
    pip install ./raymarching
```
Training requirements
- [DPT](https://github.com/isl-org/DPT). We use an off-the-shelf single-view depth estimator DPT to predict the depth for the reference image.
  ```
  git clone https://github.com/isl-org/DPT.git
  mkdir dpt_weights
  ```
  Download the pretrained model [dpt_hybrid](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt), and put it in `dpt_weights`.
- [SAM](https://github.com/facebookresearch/segment-anything). We use Segment-anything-model to obtain the foreground object mask.
- [JDiffusion](https://github.com/JittorRepos/JDiffusion). We use diffusion prior from a pretrained 2D Stable Diffusion 2.0 model. To start with, you may need download the jittor version of stable diffusion.
## Training 
### Coarse stage
We use progressive training strategy to generate a full 360° 3D geometry. Run the command and modify the workspace name `NAME`, the path of the reference image `IMGPATH` and the prompt `PROMPT` describing the image . We first optimize the scene under frontal camera views. 
```
    python main.py --workspace ${NAME} --ref_path "${IMGPATH}" --phi_range 135 225 --iters 2000 --backbone vanilla --text ${PROMPT}
```

Note that since we use the valina version of Nerf, the results will be slightly different from the pytorch version.



## Important Note
Hallucinating 3D geometry and generating novel views from a single image of general genre is a challenging task. While our method demonstrates strong capability on creating 3D from most images with a centered single object, it may still encounter difficulties in reconstructing solid geometry on complex cases. **If you encounter any bugs, please feel free to contact us.**



## Citation
If you find this code helpful for your research, please cite:
```
@InProceedings{Tang_2023_ICCV,
    author    = {Tang, Junshu and Wang, Tengfei and Zhang, Bo and Zhang, Ting and Yi, Ran and Ma, Lizhuang and Chen, Dong},
    title     = {Make-It-3D: High-fidelity 3D Creation from A Single Image with Diffusion Prior},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {22819-22829}
}
```

## Acknowledgments
This code borrows heavily from [Stable-Dreamfusion](https://github.com/ashawkey/stable-dreamfusion). 
