import cv2
import os
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, Rotate, RandomBrightnessContrast,
    HueSaturationValue, Blur, OpticalDistortion, GridDistortion, ElasticTransform
)
from skimage import io
import numpy as np


input_dir = 'data/raw/'  
output_dir = 'data/augmented/'


if not os.path.exists(output_dir):
    os.makedirs(output_dir)


augmentation = Compose([
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    Rotate(limit=45, p=0.5),
    RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    Blur(blur_limit=3, p=0.5),
    OpticalDistortion(p=0.5),
    GridDistortion(p=0.5),
    ElasticTransform(p=0.5)
])



def augment_and_save(image_file):

    image = io.imread(os.path.join(input_dir, image_file))


    augmented = augmentation(image=image)
    augmented_image = augmented['image']


    io.imsave(os.path.join(output_dir, f"aug_{image_file}"), augmented_image)



for image_file in os.listdir(input_dir):
    if image_file.endswith('.jpg'):  # Check for jpg extension
        augment_and_save(image_file)
