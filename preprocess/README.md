## Preprocess

Before training the model, you should preprocess the input image to get the corresponding depth map, segmentation mask, and 
text prompt that describing the image.



### 1. Single-view depth estimation
Depth maps are needed in our model.
In our work, we use an off-the-shelf single-view depth estimator [DPT](https://github.com/isl-org/DPT), a depth prediction model based on vision transformers, to predict the depth of the reference image. 

- Before utilizing Make-It-3D to generate a 3D object, please make sure that the requirements of DPT are met according to the [DPT project](https://github.com/isl-org/DPT).
  ```
  git clone https://github.com/isl-org/DPT.git
  cd DPT
  mkdir dpt_weights
  ```
- Download the pretrained model [dpt_hybrid](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt), and put it in `DPT/dpt_weights`.


- Estimate the depth for the input image by placing the input images in the folder ```input``` and then run: 
  ```
  python run_monodepth.py
  ```
- Then the depth map the are saved in ```preprocess/depth.png```
For example, use the DPT model on teddy-bear, you will have:

![](../demo/teddy.png) <img src="../results/preprocess/teddy/depth.png" width="378" heigh="378"/>
### 2. Image segmentation (Optional)

The input image is required to be **'RGBA'** format, where the last channel is a mask for the foreground image.

If the input image do not match the required format, you should first segment the foreground by a segmentation model (We recommend to use [Segment-anything-model](https://github.com/facebookresearch/segment-anything)).

- You can use SAM to get the foreground object mask by running:
```
git clone https://github.com/facebookresearch/segment-anything.git
```

- By placing the [checkpoints](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) in <path/to/checkpoint> and the input images in <image_or_folder>, you can then run:
```
cd segment-anything
python scripts/amg.py --checkpoint <path/to/checkpoint> --model-type <model_type> --input <image_or_folder> --output ./
```
- Then add the mask to the fourth channel of your input image to get the **'RGBA'** format, then save it in ``preprocess/your_image_name.png``


### 3. Text prompt generation (Optional)

- You can use the downloaded [BLIP](https://github.com/salesforce/BLIP) model to estimate the prompt describing the input image. (Optional)

- Or you can assign a text prompt for the input image manually.

- Save the text prompt in ```preprocess/prompt.txt```.


### 4. Move to your workspace

After getting the depth image and mask image, perform the following command to  move the depth image and text prompt into your workspace:
```
python3 mv2workspace.py --workspace ${WORKSPACE_NAME}
```

Note⚠️: Make sure to run ```mv2workspace.py``` to move the depth maps into your workspace and the foreground image into ```demo```. Ensure that the workspace name used is the same as the one used in coarse stage.
After this step, the fold ```results``` shows the following directory structure:
```
Make-It-3D/
│
├── results/
│   ├── $WORKSPACE_NAME$/
│   │    ├── preprocess/
|   |       ├── depth.png
│   │       ├── prompt.txt
│   └── ...
└── 
```

Take ```demo\astronaut.png``` as an example. After saving the 'RGBA'-format image in ```preprocess/astronaut.png``` and running mv2worspace.py to move ```depth.png``` and ```prompt.txt```, you can then 
train the coarse-stage model by running: 
```
python main.py --workspace astronaut --ref_path preprocess/astronaut.png --phi_range 135 225 --iters 10000 --backbone vanilla
```
