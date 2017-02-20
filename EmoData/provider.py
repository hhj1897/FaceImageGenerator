import h5py
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



class BASE():

    def __init__(self,
            histogram_normalization=False,
            make_grayscale = False,
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
            add_mask=False,
            ):
        '''
        '''
        self.output_size = output_size 
        self.face_size = face_size 
        self.allignment_type = allignment_type 
        # {'similarity','affine','projective'}

        self.histogram_normalization = histogram_normalization
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

    def run_pipeline(self, 
            img, 
            extract_bbox = False, 
            preprocessing = False, 
            augment = False, 
            pts = None,
            pts_out = 'pts',
            ):
        '''
        pts_out: {'pts','pts_raw','both'}
        '''

        if extract_bbox:
            img, pts, pts_raw  = bbox_extractor(
                    self.detector, 
                    self.shape_predictor, 
                    img,
                    self.output_size,
                    self.face_size,
                    self.allignment_type,
                    self.fill_mode,
                    pts = pts,
                    )
        else:
            pts = None 
            pts_raw = None 


        # normalize input to float32 [0, 1]
        img = np.float32(img)
        if np.all(img==0):
            pass
        else:
            img -= img.min()
            img /= img.max()


        if preprocessing:
            if self.make_grayscale and img.shape[-1]==3:
                img = rgb2gray(img)
                img = img[:,:,None]

            if self.histogram_normalization:
                img = np.float32(img)
                img = exposure.equalize_hist(img)

        if augment:
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


        
        if pts_out == 'pts':
            return img, pts

        elif pts_out == 'pts_raw':
            return img, pts_raw

        elif pts_out == 'both': 
            return img, pts, pts_raw


class Facial_Expressions(BASE):

    def flow_from_hdf5(self, 
            path_to_file, 
            batch_size=64,
            extract_bbox = False,
            augment = False,
            preprocessing = False,
            img_postprocessing = None,
            pts_postprocessing = None,
            lab_postprocessing = None,
            ):
        '''
        '''
        f = h5py.File(path_to_file)
        data = f['lab']

        nb_samples = data.shape[0]
        nb_batches = math.ceil(nb_samples/batch_size)

        def _make_generator(data, postprocessing=None, apply_pipeline=False):

            t0 = 0
            t1 = batch_size
            batch_counter = 0

            num_samples=data.shape[0]

            while True:

                # repeat iteration over all frames if end is reached
                t1 = min( num_samples, t1 )
                if t0 >= num_samples:
                    t0 = 0
                    t1 = batch_size

                batch = data[t0:t1]
                batch_out = []

                for img in batch:

                    # apply processing pipeline
                    if apply_pipeline:
                        img, _ = self.run_pipeline(img, extract_bbox, preprocessing, augment)

                    if postprocessing:
                        img = postprocessing(img)
                    
                    batch_out.append(img)

                batch = np.stack(batch_out)

                batch_counter+=1
                t0 += batch_size
                t1 += batch_size

                yield batch

        res_gen = {}
        res_gen['nb_samples']=nb_samples
        res_gen['nb_batches']=nb_batches
        res_gen['img_shape']=f['img'].shape[1:]
        res_gen['pts_shape']=f['pts'].shape[1:]
        res_gen['lab_shape']=f['lab'].shape[1:]
        res_gen['nb_outputs']=f['lab'].shape[1]
        res_gen['nb_classes']=f['lab'].shape[2]
        res_gen['img'] = _make_generator(f['img'], img_postprocessing, True)
        res_gen['lab'] = _make_generator(f['lab'], lab_postprocessing)
        res_gen['pts'] = _make_generator(f['pts'], lab_postprocessing)
        return res_gen

        # return _make_generator(data)

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

    def flow_from_hdf5_list(self, 
            list_of_paths, 
            batch_size=64,
            extract_bbox = False,
            preprocessing = False,
            postprocessing = None,
            augment = False,
            inputer = None,
            downscaling = 1,
            downsampling = 1,
            shuffle = False,
            one_hot = True  
            ):
        '''
        '''
        groups = {}
        with h5py.File(list_of_paths[0]) as f:
            for g in f.keys():groups[g]=[]
    
        for path in tqdm(list_of_paths):
            with h5py.File(path) as f:
                for g in f.keys():
                    if g=='img':
                        dset = f[g][::downsampling,::downscaling,::downscaling]
                    else:
                        dset = f[g][::downsampling]

                    groups[g].append(dset)

        
        for g in groups:
            groups[g] = np.vstack(groups[g])

        if inputer=='remove':
            idx_good = np.ones(groups['img'].shape[0],dtype=bool)

            for g in groups:
                idx = np.isnan(np.sum(groups[g].reshape(groups[g].shape[0],-1),1))
                idx_good[idx]=False
                idx = np.sum(groups[g].reshape(groups[g].shape[0],-1),1)==0
                idx_good[idx]=False

            for g in groups:
                groups[g] = groups[g][idx_good]

        # apply preprocessing first
        if preprocessing:
            out = []
            for i,img in enumerate(groups['img']):
                img, _ = self.run_pipeline(img, False, preprocessing=True)
                out.append(img)
            groups['img'] = np.stack(out)

        nb_samples = groups['img'].shape[0]
        nb_batches = math.ceil(nb_samples/batch_size)

        idx = np.arange(nb_samples)
        if shuffle:
            np.random.shuffle(idx)
            for g in groups:groups[g]=groups[g][idx]

        def _make_generator(data, augment=False, one_hot=False):
            t0 = 0
            t1 = batch_size
            nb_samples = data.shape[0]
            batch_counter = 0
    
            while True:

                # repeat iteration over all frames if end is reached
                t1 = min( nb_samples, t1 )
                if t0 >= nb_samples:
                    t0 = 0
                    t1 = batch_size
    
                batch = data[t0:t1]
                if one_hot==False:batch=np.argmax(batch,2)

                if augment:
                    out = []
                    # apply augmentation 
                    for i,img in enumerate(batch):
                        img, _ = self.run_pipeline(img, augment=True )
                        out.append(img)
                    batch = np.stack(out)

                batch_counter += 1
                t0 += batch_size
                t1 += batch_size

                yield batch

        res_gen = {}
        for g in groups:
            if g=='img':
                res_gen[g] = _make_generator(groups[g], augment, True)
            elif g=='lab':
                res_gen[g] = _make_generator(groups[g], False, one_hot)
            else:
                res_gen[g] = _make_generator(groups[g], False, True)

        res_gen['nb_samples']=nb_samples
        res_gen['nb_batches']=nb_batches
        res_gen['img_shape']=groups['img'].shape[1:]
        res_gen['pts_shape']=groups['pts'].shape[1:]
        res_gen['lab_shape']=groups['lab'].shape[1:]

        return res_gen
