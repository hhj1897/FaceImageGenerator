import emotion_data


fg = emotion_data.provider.Facial_Expressions(
        histogram_normalization=True,
        mean_std_normalization=True,
        greyscale=True,
        rotation_range = 10,
        width_shift_range = 0.05,
        image_size = [128, 192],
        face_size = 128,
        height_shift_range = 0.05,
        zoom_range = 0.05,
        fill_mode = 'edge',
        random_flip = True,
        path_to_shape_model = '/homes/rw2614/projects/shape_model',
        allignment_type = 'similarity',
        )


# img_tr_folder = fg.flow_from_folder(
        # './test_data/images','jpg', 
        # batch_size=3, 
        # extract_bbox=True, 
        # preprocess=True, 
        # augment=False, 
        # # add_mask=True, 
        # save_to_dir='./tmp/'
        # )

# img_tr_folder = fg.flow_from_hdf5(
        # './disfa/test.h5','img', 
        # batch_size=3, 
        # extract_bbox=False, 
        # preprocess=False, 
        # augment=False, 
        # save_to_dir='./tmp/'
        # )

import h5py
from skimage.io import imsave
f = h5py.File('./disfa/train.h5')
print(f['img'].shape)
print(f['pts'].shape)
print(f['lab'].shape)
img = f['img'][-11]
import numpy as np
print(np.argmax(f['lab'][-11],1))
img-=img.min()
img/=img.max()
imsave('test.png',img[:,:,0])

# for i in range(5):
    # next(img_tr_folder)
