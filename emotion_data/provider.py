import h5py
import dlib
import os
import numpy as np
import shutil
import glob
from skimage.color import rgb2gray, gray2rgb
from skimage.io import imsave
from skimage.io import imread
from skimage import transform, exposure, draw
from .image_augmentation import rotate, zoom, width_shift, height_shift


class Facial_Expressions():
    def __init__(self,
            histogram_normalization=False,
            mean_std_normalization=False,
            greyscale=False,
            image_size = [128,192],
            face_size = 128,
            rotation_range = 0,
            width_shift_range = 0,
            height_shift_range = 0,
            zoom_range = 0,
            fill_mode = 'edge',
            random_flip = False,
            path_to_shape_model = None,
            allignment_type = False,
            ):
        '''
        '''
        self.image_size = image_size 
        self.face_size = face_size 
        self.allignment_type = allignment_type 
        # {'similarity','affine','projective'}

        self.histogram_normalization = histogram_normalization
        self.mean_std_normalization = mean_std_normalization
        self.greyscale=greyscale

        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.zoom_range = zoom_range
        self.fill_mode = fill_mode
        self.random_flip = random_flip

        with h5py.File(os.path.dirname(__file__)+'/data/mean_shape.h5') as tmp:
            self.mean_shape = tmp['data'][::]
            self.connections = tmp['connections'][::]

        # move to origin
        self.mean_shape[0]-=self.mean_shape.mean(1)[0]
        self.mean_shape[1]-=self.mean_shape.mean(1)[1]

        # scale mean_shape to face size
        scale = max(self.mean_shape.max(1)-self.mean_shape.min(1))
        self.mean_shape*=face_size/scale

        # move back to center 
        self.mean_shape[0]+=image_size[0]/2
        self.mean_shape[1]+=image_size[1]/2

        if path_to_shape_model:
            pwd = os.path.abspath(path_to_shape_model)
            self.detector = dlib.get_frontal_face_detector()
            self.shape_predictor = dlib.shape_predictor( pwd+'/shape_predictor_68_face_landmarks.dat' )

        else:
            self.detector = None
            self.shape_predictor = None 


        if allignment_type:
            assert(path_to_shape_model!=None),\
                'allignment_type set but path_to_shape_model not assigned'

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

    def flow_from_folder(self, 
            path_to_folder, 
            suffix,
            batch_size=64,
            extract_bbox = False,
            add_mask=False,
            preprocess = False,
            augment=False,
            seed=None,
            save_to_dir=None,
            ):
 
        pwd = os.path.abspath(path_to_folder)
        files = sorted(list(glob.glob(pwd+'/*.'+suffix)))
        
        num_samples = len(files)

        t0 = 0
        t1 = batch_size
        num_batch = 0

        while True:
            num_batch += 1
            t1 = min(num_samples,t1)
            if t0>=num_samples:
                t0=0
                t1 = batch_size

            batch_img = []
            batch_pts = []
            for i in range(t0,t1):
                img = imread(files[i])
                if len(img.shape)==2:img=gray2rgb(img)

                img, pts  = self.process_image(img, extract_bbox, preprocess, augment, add_mask)
                batch_img.append(img)
                batch_pts.append(pts)

            if save_to_dir!=None:
                for sample, [img, pts] in enumerate(zip(batch_img, batch_pts)):
                    if add_mask:
                        if np.all(img==False):img=np.zeros(self.image_size[::-1])
                        if np.all(pts==False):pts=np.self.mean_shape.T
                        img = self._add_mask(img,pts)
                    out_path = save_to_dir+'/'+str(num_batch).zfill(5)+'_'+str(sample+1).zfill(5)+'.jpg'
                    self._save_to_dir(img, out_path)

            t0 += batch_size
            t1 += batch_size

            yield np.stack(batch_img), np.stack(batch_pts)

    def process_image(self, img, extract_bbox, preprocess, augment):
        '''
        '''
        if extract_bbox:
            img, pts = self._align_face_and_extract_bbox(img)
        else:
            pts = self.mean_shape.T

        if preprocess:
            img = self._preprocess(img) 

        if augment:
            img = self._augmentation(img)

        return img, pts

    def _save_to_dir(self, img, pwd):
        '''
        '''
        if np.all(img==False):img=np.zeros(self.image_size[::-1])
        img = np.float32(img)
        img-=img.min()
        img/=img.max()
        imsave(pwd,img)

    def _align_face_and_extract_bbox(self, img, points='all'):
        p = {   'inner' : np.arange(17,68),
                'outer' : np.arange(0,27),
                'stable': np.array([36,39,42,45,33]),
                'Stable': np.array([19, 22, 25, 28, 10, 11, 12, 13, 14, 15, 16, 17, 18])+17,
                'all'   : np.arange(68)}

        tmp = np.float32(np.copy(img))
        tmp -= tmp.min()
        tmp /= tmp.max()
        tmp = np.uint8(tmp*255)
        rect = self.detector(tmp, 0)

        # if tracking fails: return zeros 
        if len(rect)==0:
            return np.zeros(self.image_size[::-1]), self.mean_shape.T

        bbox = rect[0]
        shape = self.shape_predictor(tmp, bbox)
        pts = np.vstack([ [p.x, p.y] for p in shape.parts() ])

        # if self.allignment_type==None:
        src = self.mean_shape.T[p[points]]
        dst = pts[p[points]]
        trans = transform.estimate_transform(self.allignment_type, src, dst)

        min_val, max_val = img.min(), img.max()
        img = (img-min_val)/max_val
        img = transform.warp( img, inverse_map = trans, mode=self.fill_mode, output_shape=self.image_size[::-1] )
        img = (img*max_val)+min_val

        pts = trans.inverse(pts)

        return img, pts

    def _add_mask(self, img, pts):
        '''
        '''
        if len(img.shape)==2:img=gray2rgb(img)

        radius = max(pts.max(0)-pts.min(0))/60
        max_val = np.max(img)
        for x,y in np.int16(pts):
            rr, cc = draw.circle(y, x, radius)
            try:
                img[rr, cc, 0] = max_val
                img[rr, cc, 1] = max_val
                img[rr, cc, 2] = 0
            except IndexError:
                pass

        for p0, p1 in self.connections:
            x = np.int16(pts[p0])
            y = np.int16(pts[p1])

            rr, cc, val = draw.line_aa(x[1],x[0],y[1],y[0])
            try:
                img[rr, cc, 1] = max_val
            except IndexError:
                pass


        return img
        
    def _preprocess(self, img):
        '''
        '''
        trans = transform.SimilarityTransform(translation=0)

        if self.greyscale:
            img = rgb2gray(img)

        img = np.float32(img)

        if self.histogram_normalization:
            img = exposure.equalize_hist(img)

        if self.mean_std_normalization:
            img -= img.mean()
            img /= img.std()

        return img

    def _augmentation(self, img):
        # compute random rotation
        shape = img.shape

        deg = np.random.uniform(-self.rotation_range, self.rotation_range)
        trans = rotate(shape, deg)

        # compute random zooming
        z = np.random.uniform(-self.zoom_range, self.zoom_range)
        trans += zoom(shape, z)

        # compute random shifting
        shift = np.random.uniform(-self.width_shift_range, self.width_shift_range)
        trans += width_shift(shape, shift)

        # compute random shifting
        shift = np.random.uniform(-self.height_shift_range, self.height_shift_range)
        trans += height_shift(shape, shift)

        if self.random_flip:
            if np.random.rand()>0.5:
                img=img[:,::-1]

        # scale image, apply transofrmation, scale it back
        # values between -1 and 1 are required for the transformation 
        min_val, max_val = img.min(), img.max()
        img = (img-min_val)/(max_val-min_val)
        img = transform.warp(img, trans, mode=self.fill_mode)
        img = (img*(max_val-min_val))+min_val
        
        return img 
