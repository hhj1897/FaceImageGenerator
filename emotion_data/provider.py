import h5py
import dlib
import os
import numpy as np
import shutil
from skimage.io import imsave
from skimage import transform, exposure 
from .image_augmentation import rotate, zoom, width_shift, height_shift



class Facial_Expressions():
    def __init__(self,
            histogram_normalization=False,
            mean_std_normalization=False,
            allignment_type = False,
            downscaling = 1,
            rotation_range = 0,
            width_shift_range = 0,
            height_shift_range = 0,
            zoom_range = 0,
            fill_mode = 'edge',
            random_flip = False,
            path_to_shape_model = None,
            ):
        '''
        '''
        self.histogram_normalization = histogram_normalization
        self.mean_std_normalization = mean_std_normalization
        self.downscaling = downscaling
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.zoom_range = zoom_range
        self.fill_mode = fill_mode
        self.random_flip = random_flip
        self.allignment_type = allignment_type 
        # {'similarity','affine','projective'}

        self.shape_predictor = None
        self.detector = None
        self.mean_shape = None

        if allignment_type:
            assert(path_to_shape_model!=None),\
                'allignment_type set but path_to_shape_model not assigned'
            pwd = os.path.abspath(path_to_shape_model)

            self.detector = dlib.get_frontal_face_detector()
            self.shape_predictor = dlib.shape_predictor( pwd+'/shape_predictor_68_face_landmarks.dat' )

            with h5py.File(pwd+'/mean_shape.h5') as tmp:
                self.mean_shape = tmp['data'][::]

    def flow_from_hdf5(self, 
            path_to_hdf5_file, 
            group_name, 
            batch_size=64,
            augment=False,
            seed=None,
            save_to_dir=None,
            ):
        '''
        '''
        f = h5py.File(path_to_hdf5_file)
        data = f[group_name]

        # scale mean shape to input size (will be needed for the similarity transformation)
        if self.allignment_type:

            sample_size = data[0, ::self.downscaling, ::self.downscaling, 0].shape
            self.mean_shape*=max(sample_size)
            self.mean_shape[0]+=sample_size[1]/2-self.mean_shape.mean(1)[0]
            self.mean_shape[1]+=sample_size[0]/2-self.mean_shape.mean(1)[1]

        t0 = 0
        t1 = batch_size
        num_batch = 0

        while True:
            num_batch += 1
            t1 = min(data.shape[0],t1)
            if t0>=data.shape[0]:
                t0=0
                t1 = batch_size


            batch = data[t0:t1, ::self.downscaling, ::self.downscaling]
            batch = np.stack(self.preprocess(i, augment, seed) for i in batch)

            if save_to_dir!=None:
                for sample,img in enumerate(batch):
                    img = np.float32(img)
                    img-=img.min()
                    img/=img.max()
                    out_path = save_to_dir+'/'+str(num_batch).zfill(5)+'_'+str(sample+1).zfill(5)+'.jpg'
                    imsave(out_path,img[:,:,0])

            t0 += batch_size
            t1 += batch_size

            yield batch

    def preprocess(self, img, augment=False, seed=None):
        '''
        '''
        img = np.float32(img)


        if self.histogram_normalization:
            img = exposure.equalize_hist(img)

        if self.mean_std_normalization:
            img -= img.mean()
            img /= img.std()

        trans = transform.SimilarityTransform(translation=0)

        if self.allignment_type:
            tmp = np.copy(img)
            tmp -= tmp.min()
            tmp /= tmp.max()
            tmp = np.uint8(tmp*255)[:,:,0]
            rects = self.detector(tmp, 0)
            if len(rects):
                shape = self.shape_predictor(tmp, rects[0])
                landmarks = np.vstack([ [p.x, p.y] for p in shape.parts() ]).T
                trans += transform.estimate_transform(self.allignment_type, self.mean_shape.T, landmarks.T)

        if augment:

            # compute random rotation
            deg = np.random.uniform(-self.rotation_range, self.rotation_range)
            trans += rotate(img, deg)

            # compute random zooming
            zoom = np.random.uniform(-self.zoom_range, self.zoom_range)
            trans += zoom(img, zoom)

            # compute random shifting
            shift = np.random.uniform(-self.width_shift_range, self.width_shift_range)
            trans += width_shift(img, shift)

            # compute random shifting
            shift = np.random.uniform(-self.height_shift_range, self.height_shift_range)
            trans += height_shift(img, shift)


        # scale image, apply transofrmation, scale it back
        # values between -1 and 1 are required for the transofrmation
        if augment or self.allignment_type:
            min_val, max_val = img.min(), img.max()
            img = (img-min_val)/max_val
            img = transform.warp(img, trans, mode=self.fill_mode)
            img = (img*max_val)+min_val

        if self.random_flip:
            if np.random.rand()>0.5:
                img=img[:,::-1]
        
        return img 
