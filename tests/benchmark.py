import EmoData 
import numpy as np
from tqdm import tqdm
from time import time
import h5py
import numpy as np
from skimage.color import gray2rgb

path_to_shape_model = '/homes/rw2614/projects/shape_model/shape_predictor_68_face_landmarks.dat'

fg = EmoData.provider.Facial_Expressions(
    histogram_normalization = True,
    mean_std_normalization = True,
    make_grayscale = True,
    path_to_shape_model = path_to_shape_model,
    # output_size = [128,192],
    # face_size = 128,
    rotation_range = 90,
    width_shift_range = 0.05,
    height_shift_range = 0.05,
    zoom_range = 0.05,
    )

def process(dset):

    out = []
    for i in range(dset.shape[0]):
        img=gray2rgb(dset[i])
        tmp, pts = fg.run_pipeline(dset[i], extract_bbox=False, preprocessing=True, augment=True)
        out.append(tmp)
    out = np.stack(out) 

    return dset


batch_size = 10
samples_per_second = []
with h5py.File('/homes/rw2614/data/NEW/similarity_240_160_3/fera2015_train.h5') as f:
    dset = f['img']
    t0 = time()

    for i in tqdm(range(20)):
        i0 = i*batch_size
        i1 = (i+1)*batch_size
        batch = dset[i0:i1]
        process(batch)
        t1 = time()
        samples_per_second_in_batch = 1/((t1-t0)/batch_size)
        samples_per_second.append(samples_per_second_in_batch)
        t0 = time()

print('iteration per second:')
print(np.mean(samples_per_second))
