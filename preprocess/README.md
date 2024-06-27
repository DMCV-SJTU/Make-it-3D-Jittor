## Preprocess

Before training the model, you should preprocess the input image to get the corresponding depth map, segmentation mask, and 
text prompt that describing the image.

### 0. Download the following models

- [DPT](https://github.com/isl-org/DPT). We use an off-the-shelf single-view depth estimator DPT to predict the depth for the reference image. DPT is a depth prediction model based on vision transformers. Before utilizing Make-It-3D to generate a 3D object, please make sure that the requirements of DPT are met according to the [DPT project](https://github.com/isl-org/DPT).
  ```
  git clone https://github.com/isl-org/DPT.git
  mkdir dpt_weights
  ```
  Download the pretrained model [dpt_hybrid](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt), and put it in `dpt_weights`.
- [SAM](https://github.com/facebookresearch/segment-anything). We use Segment-anything-model to obtain the foreground object mask.
- [JDiffusion](https://github.com/JittorRepos/JDiffusion). We use diffusion prior from a pretrained 2D Stable Diffusion 2.0 model. To start with, you may need download the jittor version of stable diffusion.
- [Stable Diffusion 2.0]() You can dowlond the [weights](https://huggingface.co/stabilityai/stable-diffusion-2-base/tree/main) for sd2 into the sd2 folder.
- [clip-b16] 

### 1. Single-view depth estimation

You can use the downloaded DPT model to estimate the depth for input image:

  ```
  python3 dpt.py --input=$path_to_input_image
  ```
For example, use the DPT model on teddy-bear:
  ```
  python3 dpt.py --input=../demo/teddy.png
  ```
![](../demo/1_ref_depth_mask.png)
### 2. Image Segmentation

You can use the downloaded SAM model to estimate the segmentation mask for the input image:
  ```
  python3 sam.py --input=$path_to_input_image
  ```

### 3. Prompt generation(Optional)
You can use the downloaded BLIP model to estimate the prompt describing the input image:
  ```
  python3 blip.py --input=$path_to_input_image
  ```

