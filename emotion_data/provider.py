import h5py
import dlib
import os
import numpy as np
import glob
from skimage.color import rgb2gray, gray2rgb
from skimage.io import imread
from skimage import transform, exposure
from .image_augmentation import rotate, zoom, width_shift, height_shift
from .image_processing import bbox_extractor, save_image, add_landmarks_to_img 


class Facial_Expressions():
    def __init__(self,
            histogram_normalization=False,
            mean_std_normalization=False,
            make_grayscale = False,
            output_size = [128,192],
            face_size = 128,
            rotation_range = 10,
            width_shift_range = 0.05,
            height_shift_range = 0.05,
            zoom_range = 0.05,
            fill_mode = 'edge',
            random_flip = True,
            path_to_shape_model = None,
            allignment_type = 'similarity',
            add_mask=False,
            ):
        '''
        '''
        self.output_size = output_size 
        self.face_size = face_size 
        self.allignment_type = allignment_type 
        # {'similarity','affine','projective'}

        self.histogram_normalization = histogram_normalization
        self.mean_std_normalization = mean_std_normalization
        self.make_grayscale = make_grayscale

        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.zoom_range = zoom_range
        self.fill_mode = fill_mode
        self.random_flip = random_flip

        self.add_mask = add_mask


        if path_to_shape_model:
            pwd = os.path.abspath(path_to_shape_model)
            self.detector = dlib.get_frontal_face_detector()
            self.shape_predictor = dlib.shape_predictor( pwd )
        else:
            self.detector = None
            self.shape_predictor = None 

    def flow_from_hdf5(self, 
            path_to_file, 
            group_name,
            batch_size=64,
            extract_bbox = False,
            preprocessing = False,
            postprocessing = None,
            augment=False,
            save_to_dir=None,
            ):
        '''
        '''
        f = h5py.File(path_to_file)
        data = f[group_name]

        t0 = 0
        t1 = batch_size
        num_samples = data.shape[0]
        batch_counter = 0

        while True:

            # repeat iteration over all frames if end is reached
            t1 = min( num_samples, t1 )
            if t0 >= num_samples:
                t0 = 0
                t1 = batch_size

            batch = data[t0:t1]
            batch_out = []

            for sample, img in enumerate(batch):

                # apply processing pipeline
                img, _ = self.run_pipeline(img, extract_bbox, preprocessing, augment)
                if postprocessing:img = postprocessing(img)
                batch_out.append(img)

                if save_to_dir!=None:
                    out_path = save_to_dir+'/'+str(batch_counter).zfill(5)+'_'+str(sample+1).zfill(5)+'.jpg'
                    save_image(img, out_path)

            batch_counter+=1
            t0 += batch_size
            t1 += batch_size

            yield np.stack(batch_out)

    def flow_from_folder(self, 
            path_to_folder, 
            suffix,
            batch_size=64,
            extract_bbox = False,
            preprocessing = False,
            augment=False,
            save_to_dir=None,
            ):
 
        pwd = os.path.abspath(path_to_folder)
        files = sorted(list(glob.glob(pwd+'/*.'+suffix)))
        
        t0 = 0
        t1 = batch_size
        num_samples = len(files)
        batch_counter = 0

        while True:

            # repeat iteration over all frames if end is reached
            t1 = min( num_samples, t1 )
            if t0 >= num_samples:
                t0 = 0
                t1 = batch_size

            batch_img = []
            batch_pts = []

            for i, sample in enumerate(range(t0,t1)):
                img = imread(files[i])

                # add rgb dimension, if image is grayscale
                if len(img.shape)==2:img=gray2rgb(img)

                # apply processing pipeline
                img, pts  = self.run_pipeline(img, extract_bbox, preprocessing, augment)
                batch_img.append(img)
                batch_pts.append(pts)

                if save_to_dir!=None:
                    out_path = save_to_dir+'/'+str(batch_counter).zfill(5)+'_'+str(sample+1).zfill(5)+'.jpg'
                    save_image(img, out_path)

            batch_counter+=1
            t0 += batch_size
            t1 += batch_size

            if extract_bbox:
                yield np.stack(batch_img), np.stack(batch_pts)
            else:
                yield batch_img, batch_pts

    def run_pipeline(self, img, extract_bbox, preprocessing, augment):
        '''
        '''
        if extract_bbox:
            img, pts  = bbox_extractor(
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

        if preprocessing:

            if self.make_grayscale and img.shape[-1]==3:
                img = rgb2gray(img)
            img = np.float32(img)

            if self.histogram_normalization:
                img = exposure.equalize_hist(img)

            if self.mean_std_normalization:
                img -= img.mean()
                img /= img.std()

        if augment:
            # compute random rotation
            deg = np.random.uniform(-self.rotation_range, self.rotation_range)
            trans = rotate(img.shape, deg)

            # compute random zooming
            z = np.random.uniform(-self.zoom_range, self.zoom_range)
            trans += zoom(img.shape, z)

            # compute random shifting
            shift = np.random.uniform(-self.width_shift_range, self.width_shift_range)
            trans += width_shift(img.shape, shift)

            # compute random shifting
            shift = np.random.uniform(-self.height_shift_range, self.height_shift_range)
            trans += height_shift(img.shape, shift)

            if self.random_flip:
                if np.random.rand()>0.5:
                    img=img[:,::-1]

            # scale image, apply transofrmation, scale it back
            # values between -1 and 1 are required for the transformation 

            min_val, max_val = img.min(), img.max()
            img = (img-min_val)/(max_val-min_val)
            img = transform.warp(img, trans, mode=self.fill_mode)
            img = (img*(max_val-min_val))+min_val

            # pts = trans.inverse(pts)

        if self.add_mask:
            img = add_landmarks_to_img(img,pts)

        return img, pts
