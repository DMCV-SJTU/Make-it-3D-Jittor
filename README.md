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
- [X] **Release refine stage training code** 
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

Please download the requirement folds from [here](https://drive.google.com/drive/folders/16vN86aBc1XLsbIHL0tMpgX9jcgyUdrir?usp=drive_link) The directory structure of downloaded fold is as following:
```
makeit3d_requirement/
│
├── diffuser_jittor/
│   ├── setup.py
│   └── ...
|   
├── JDiffusion/
│   ├── setup.py
│   └── ...
| 
├── jtorch/
│   ├── setup.py
|   └── ...
|
├── transformers_jittor/
│   ├── setup.py
|   └── ...
```
In the same directory as each setup.py file mentioned above, run the following command:
```
pip install -e .
```
Other dependencies:
```
pip install -r requirements.txt
```

### Preprocess before training 
Before training the model, you should preprocess the input image to get the corresponding depth map, segmentation mask, and 
text prompt that describing the image. More details can be referred
in ```./preporcess/README.md```.

## Training 
### Coarse stage
We use progressive training strategy to generate a full 360° 3D geometry. Run the command and modify the workspace name `NAME`, the path of the reference image `IMGPATH` and the prompt `PROMPT` describing the image . We first optimize the scene under frontal camera views. 
```
python main.py --workspace ${NAME} --ref_path "${IMGPATH}" --phi_range 135 225 --iters 10000 --backbone vanilla --text ${PROMPT}
```
For exeample, you can implement your command as:
```
python main.py --workspace teddy --ref_path demo/teddy.png --phi_range 135 225 --iters 2000 --backbone vanilla --text "a teddy bear"
```
Note that since we use the valina version of Nerf, the results will be slightly different from the pytorch version.

### Refine stage
We have proposed an example for refine stage. Before the refine stage training, you should download [following examples](https://drive.google.com/drive/folders/1hy88cet39yYM_WjF94b3rHF4XCrpgH6m?usp=sharing) into your workspace. Make sure the downloaded files are placed in the following directory structure: 
```
Make-It-3D/
│
├── results/
│   ├── $WORKSPACE_NAME$/
│   │    ├── mvimg/
|   |       ├── df_epxxx_000_depth.png
│   │       ├── df_epxxx_000_mask.png
│   │       ├── df_epxxx_000_normal.png
│   │       ├── df_epxxx_000_rgb.png
|   |       ├── df_epxxx_poses.npy
│   │       └── ...  
│   │    ├── refine/
│   └── ...
└── ...
```

#### Teddy bear
You can easily refine this teddy bear texture as following guidance:
```
python main.py --workspace ${WORKSPACE_NAME} --ref_path "demo/teddy.png" --phi_range 0 90 --fovy_range 50 70 --fov 60 --refine --refine_iter 3000 --backbone vanilla --text "a teddy bear"
```

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
