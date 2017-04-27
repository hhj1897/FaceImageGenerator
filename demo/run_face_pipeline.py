from FaceImageGenerator.image_pipeline import FACE_pipeline 
import os
import numpy as np
from skimage.io import imread, imsave
pwd = os.path.dirname(os.path.abspath(__file__))

pip = FACE_pipeline(
        output_size = [128,128],
        face_size = 100,
        histogram_normalization=True,
        grayscale=False,
        rotation_range = 5.,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        zoom_range = 0.1,
        gaussian_range = 5,
        fill_mode = 'edge',
        random_flip = True,
        )

img = imread(pwd+'/input.jpg')

# apply face detecton and allignment
out, pts, pts_raw = pip.transform(
        img, 
        face_detect=True, 
        preprocessing=False, 
        augmentation=False,
        )
imsave('face.jpg',out)

# apply face detecton, allignment and preprocessing
# (Use this pipeline for your CNN test data)
out, pts, pts_raw = pip.transform(
        img, 
        face_detect=True, 
        preprocessing=True, 
        augmentation=False,
        )
imsave('test.jpg',out)

# apply face detecton, allignment, preprocessing and data augmentation
# (Use this pipeline for your CNN training data)
out, pts, pts_raw = pip.transform(
        img, 
        face_detect=True, 
        preprocessing=True, 
        augmentation=True,
        )
imsave('train.jpg',out)
