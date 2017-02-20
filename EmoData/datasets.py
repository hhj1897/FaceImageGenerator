import numpy as np
import glob
from . import provider



def _to_class_vector(x):
    return np.argmax(x,1)


def fera2015(batch_size=32, augment=False, preprocessing=True, rot=0, scale=0, gray=True, validation=True):
    '''
    '''
    fg = provider.Facial_Expressions(
        make_grayscale = gray,
        rotation_range = rot,
        width_shift_range = scale,
        height_shift_range = scale,
        zoom_range = scale,
        )

    TR = fg.flow_from_hdf5('/homes/rw2614/data/NEW/similarity_240_160_3/fera2015_tr.h5',
            batch_size = batch_size,
            preprocessing = True,
            augment = True,
            # lab_postprocessing = _to_class_vector,
            )

    TE = fg.flow_from_hdf5('/homes/rw2614/data/NEW/similarity_240_160_3/fera2015_te.h5',
            batch_size = batch_size,
            preprocessing = True,
            augment = False,
            # lab_postprocessing = _to_class_vector,
            )

    return TR, TE
