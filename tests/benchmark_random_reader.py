# import EmoData 
import numpy as np
from tqdm import tqdm
from time import time
import h5py
import numpy as np
from skimage.color import gray2rgb


import glob
h5_file_list = []
for f in sorted(list(glob.glob('/homes/rw2614/data/NEW/similarity_240_160_3/fera2015/*.h5'))):
    h5_file_list.append(h5py.File(f))
    print(f)

idx = []
for f in h5_file_list:
    idx.extend( 
            zip(range(f['img'].shape[0]), [f]*f['img'].shape[0]),
            )
    # f.close()

from random import shuffle
shuffle(idx)
# print(idx[0])
for frame, h5_file in idx:
    print(h5_file['img'][frame][::].shape)
    # print(h5_file[frame]['img'][::].shape)
