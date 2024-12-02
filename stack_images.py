from PIL import Image
from glob import glob
import torch
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import os

def stack_images(rgb_image_paths):
    stacked_images = []
    for path in rgb_image_paths:
        rgb_image = Image.open(path)
        nir_path = path.replace("leftImg8bit", "nir").replace("rgb", "nir")
        nir_image = Image.open(nir_path).convert('L')
        nir_image_tensor = ToTensor()(nir_image)
        rgb_image_tensor = ToTensor()(rgb_image)
        stack_image = torch.cat((nir_image_tensor, rgb_image_tensor), dim=0)
        print(stack_image.shape)
        stack_image = np.reshape(stack_image, (stack_image.shape[1], stack_image.shape[2], 4))
        # Create output directory structure
        output_dir = os.path.dirname(path).replace("leftImg8bit", "stacked")
        os.makedirs(output_dir, exist_ok=True)

        # Save the stacked image
        output_path = os.path.join(output_dir, os.path.basename(path)).replace("rgb","stack")
        plt.imsave(output_path, stack_image)
        stacked_images.append(stack_image)
    return stacked_images
    
train_rgb_image_paths = glob("/ssd_scratch/cvit/furqan.shaik/IDDAW_final/test/SNOW/leftImg8bit/**/*.png")

stacked_images = stack_images(train_rgb_image_paths)
print(len(stacked_images))
# test_rgb_image_paths = glob("/ssd_scratch/cvit/furqan.shaik/IDDE_final_with_test_split/test/SNOW/leftImg8bit/**/*.png")
# stack_images(test_rgb_image_paths)