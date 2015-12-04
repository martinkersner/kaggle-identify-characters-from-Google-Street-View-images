#!/usr/bin/python

# Martin Kersner, m.kersner@gmail.com
# 2015/12/04

# Modified script from kaggle competition called First steps with Julia

import glob
from skimage.transform import resize
from skimage.io import imread, imsave
import numpy as np
import os


def get_filename(path):
    return os.path.split(path)[-1]

def create_dir(dir_name, suffix):
    new_dir_name = dir_name + suffix

    if not os.path.exists(new_dir_name):
       os.makedirs(new_dir_name)
       return new_dir_name
    else:
        return None
        
def resize_images(dir_name, suffix, W, H):
    new_dir_name = create_dir(dir_name, suffix)
    if new_dir_name == None:
        return

    img_files = glob.glob(dir_name + "/*")

    for i, file_name in enumerate(img_files):
        img = imread(file_name)
        img_resized = resize(img, (W,H), preserve_range=True).astype(np.uint8)
        new_file_name = os.path.join(new_dir_name, get_filename(file_name))
        imsave (new_file_name, img_resized)

if __name__ == "__main__":
    train_dir = "train_png"
    test_dir  = "test_png" 
    suffix = "_resized256"

    W = 256
    H = 256

    resize_images(train_dir, suffix, W, H)
    resize_images(test_dir, suffix, W, H)
