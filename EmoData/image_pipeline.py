import dlib
import os
import numpy as np
import glob
import math
from tqdm import tqdm
from skimage.color import rgb2gray, gray2rgb
from skimage.io import imread
from skimage import transform, exposure
from .image_augmentation import rotate, zoom, width_shift, height_shift
from .image_processing import bbox_extractor, save_image, add_landmarks_to_img 

from multiprocessing import Pool




class FACE_pipeline():

    def __init__(self,
            histogram_normalization=False,
            grayscale = False,
            output_size = [160,240],
            face_size = 160,
            rotation_range = 10,
            width_shift_range = 0.05,
            height_shift_range = 0.05,
            zoom_range = 0.05,
            fill_mode = 'edge',
            random_flip = True,
            path_to_shape_model = None,
            allignment_type = 'similarity',
            ):
        '''
        '''
        self.output_size = output_size 
        self.face_size = face_size 
        self.allignment_type = allignment_type 
        # {'similarity','affine','projective'}

        self.histogram_normalization = histogram_normalization
        self.grayscale = grayscale

        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.zoom_range = zoom_range
        self.fill_mode = fill_mode
        self.random_flip = random_flip

        self.shape_predictor = None
        self.path_to_shape_model = path_to_shape_model

        self.nb_workers=12

    def transform(self, 
            img, 
            face_detect = False, 
            preprocessing = False, 
            augmentation = False, 
            ):

        pts = None
        if face_detect:

            # load detector
            if self.shape_predictor==None:
                self.detector = dlib.get_frontal_face_detector()

                if self.path_to_shape_model!=None:
                    pwd = os.path.abspath(self.path_to_shape_model)
                else:
                    pwd = os.path.dirname(os.path.abspath(__file__))
                    pwd = pwd+'/data/shape_predictor_68_face_landmarks.dat'
                self.shape_predictor = dlib.shape_predictor( pwd )

            img, pts, pts_raw  = bbox_extractor(
                    self.detector, 
                    self.shape_predictor, 
                    img,
                    self.output_size,
                    self.face_size,
                    self.allignment_type,
                    self.fill_mode,
                    )
        else:
            pts = None 


        # normalize input to float32 with values between 0 and 1
        img = np.float32(img)
        if np.all(img==0):
            pass
        else:
            img -= img.min()
            img /= img.max()

        if preprocessing:
            if self.grayscale and img.shape[-1]==3:
                img = rgb2gray(img)
                img = img[:,:,None]

            if self.histogram_normalization:
                img = np.float32(img)
                img = exposure.equalize_hist(img)

        if augmentation:
            # compute random rotation
            deg = np.random.uniform(-self.rotation_range, self.rotation_range)
            trans = rotate(img.shape, deg)

            # # compute random zooming
            z = np.random.uniform(-self.zoom_range, self.zoom_range)
            trans += zoom(img.shape, z)

            # # compute random shifting
            shift = np.random.uniform(-self.width_shift_range, self.width_shift_range)
            trans += width_shift(img.shape, shift)

            # compute random shifting
            shift = np.random.uniform(-self.height_shift_range, self.height_shift_range)
            trans += height_shift(img.shape, shift)

            if self.random_flip:
                if np.random.rand()>0.5:
                    img=img[:,::-1]

            img_tr = transform.warp(img, trans, mode=self.fill_mode)
            # if transformation is valid, update image

            if not (np.all(img==0) or np.any(np.isnan(img))):
                img = img_tr 
                if pts!=None:
                    pts = trans.inverse(pts)

        return img, pts

    @staticmethod
    def _sample_transform(arg_dict):
        fun = arg_dict['fun']
        args = arg_dict['args']
        kwargs = arg_dict['kwargs']
        img, pts = fun(*args, **kwargs)
        return img, pts 

    def batch_transform(self, batch, *args, **kwargs):

        arg_list = []
        for sample in batch:
            arg_dict = {}
            arg_dict['fun']=self.transform
            arg_dict['args']=[sample, *args]
            arg_dict['kwargs']=kwargs
            arg_list.append(arg_dict)

        P = Pool(self.nb_workers)
        out = P.map(self._sample_transform,arg_list)

        out_img = []
        out_pts = []
        for img, pts in out:
            out_img.append(img)
            out_pts.append(pts)
        out_img = np.stack(out_img)
        out_pts = np.stack(out_pts)
        return out_img, out_pts

