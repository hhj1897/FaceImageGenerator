import numpy as np
import glob
from .provider import flow_from_hdf5



def _to_class_vector(x):
    return np.argmax(x,1)


def fera2015(batch_size=32):
    '''
    '''
    TR = flow_from_hdf5('/homes/rw2614/data/NEW/similarity_160_240/fera2015_tr.h5')
    TE = flow_from_hdf5('/homes/rw2614/data/NEW/similarity_160_240/fera2015_te.h5')

        # make_grayscale = True,
        # rotation_range = rot,
        # width_shift_range = scale,
        # height_shift_range = scale,
        # random_flip = flip,
        # zoom_range = scale,
        # )


    return TR, TE

def pain(batch_size=32, augment=False, rot=5, scale=0.05, flip=True):
    '''
    '''
    fg = provider.image_data(
        make_grayscale = True,
        rotation_range = rot,
        width_shift_range = scale,
        height_shift_range = scale,
        random_flip = flip,
        zoom_range = scale,
        )

    TR = fg.flow_from_hdf5('/homes/rw2614/data/NEW/similarity_160_240/pain_tr.h5',
            batch_size = batch_size,
            preprocessing = True,
            augment = augment,
            )

    TE = fg.flow_from_hdf5('/homes/rw2614/data/NEW/similarity_160_240/pain_te.h5',
            batch_size = batch_size,
            preprocessing = True,
            augment = False,
            )


    return TR, TE
