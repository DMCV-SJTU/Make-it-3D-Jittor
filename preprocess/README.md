## Preprocess

Before training the model, you should preprocess the input image to get the corresponding depth map, segmentation mask, and 
text prompt that describing the image.

### 1. Single-view depth estimation

- [DPT](https://github.com/isl-org/DPT). We use an off-the-shelf single-view depth estimator DPT to predict the depth for the reference image. DPT is a depth prediction model based on vision transformers. Before utilizing Make-It-3D to generate a 3D object, please make sure that the requirements of DPT are met according to the [DPT project](https://github.com/isl-org/DPT).
  ```
  git clone https://github.com/isl-org/DPT.git
  mkdir dpt_weights
  ```
- Download the pretrained model [dpt_hybrid](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt), and put it in `dpt_weights`.


- Estimate the depth for the input image by placing the input images in the folder ```input``` and then run: 
  ```
  python run_monodepth.py
  ```
For example, use the DPT model on teddy-bear, you will have:

![](../demo/teddy.png) <img src="../demo/1_ref_depth_mask.png" width="378" heigh="378"/>
### 2. Image segmentation

- [SAM](https://github.com/facebookresearch/segment-anything). We use Segment-anything-model to obtain the foreground object mask. 

- By placing the [checkpoints](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) in <path/to/checkpoint> the input images in <image_or_folder>, you can then run:
```
python scripts/amg.py --checkpoint <path/to/checkpoint> --model-type <model_type> --input <image_or_folder> --output ./
```

### 3. Text prompt generation

- You can use the downloaded [BLIP](https://github.com/salesforce/BLIP) model to estimate the prompt describing the input image. (Optional)

- Or you can assign a text prompt for the input image manually.

- Save the text prompt in ```preprocess/prompt.txt```.


[//]: # (### 4. Move to your workspace)

[//]: # (After getting the depth image and mask image, perform the following command to  move all the results into your workspace:)

[//]: # (  ```)

[//]: # (  python3 preprocess.py)

[//]: # (  ```)