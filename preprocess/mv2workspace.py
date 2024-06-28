import PIL.Image as Image
import numpy as np
import argparse
import os


def convert_rgb2rgba(input_path, mask_path, output_path):
    img = Image.open(input_path).convert('RGBA')
    mask = Image.open(mask_path).convert('L')
    img = np.array(img)
    mask = np.array(mask)
    img[:, :, 3] = mask
    img = Image.fromarray(img).convert('RGBA')
    img.save(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--mask', type=str, default=None)
    parser.add_argument('--output', type=str, default='preprocess')
    if opt.input  is None or opt.mask is None:
        print("there is no image or mask")
        sys.exit(0)
    filename = os.path.basename(opt.input)
    save_path = os.path.join(opt.output, filename)
    convert_rgb2rgba(opt.input, opt.mask, save_path)
