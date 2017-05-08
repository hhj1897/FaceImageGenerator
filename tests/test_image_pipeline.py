from FaceImageGenerator.image_pipeline import FACE_pipeline 
import os
import numpy as np
from skimage.io import imread, imsave
pwd = os.path.dirname(os.path.abspath(__file__))

img = imread(pwd+'/data/images/test_07.jpg')

class testcase:

    def test_run_empty(self):
        pip = FACE_pipeline()
        out, pts, pts_raw = pip.transform(img) 
        assert(img.shape==out.shape) 
        assert(pts==None) 

    def test_grayscale(self):
        pip = FACE_pipeline(
                grayscale=True
                )
        out, pts, pts_raw = pip.transform(img, preprocessing=True)
        assert(img.shape[:2]==out.shape[:2])
        assert(out.shape[-1]==1)
        assert(pts==None)

    def test_normalization(self):
        pip = FACE_pipeline(
                histogram_normalization = True,
                )
        out, pts, pts_raw = pip.transform(img, preprocessing=True)
        assert(img.shape==out.shape)
        assert(out.max()<=255)
        assert(out.min()>=0)
        assert(pts==None)

        tmp = np.zeros_like(img)
        out, pts, pts_raw = pip.transform(tmp, preprocessing=True)
        assert(tmp.shape==out.shape)
        assert(np.all(out==0.5))
        assert(pts==None)

    def test_standarzation(self):
        pip = FACE_pipeline(
                standardisation = True,
                histogram_normalization = True,
                )
        out, pts, pts_raw = pip.transform(img, preprocessing=True)
        assert(img.shape==out.shape)
        assert(out.max()>=0)
        assert(out.min()<=0)
        assert(pts==None)

        tmp = np.zeros_like(img)
        out, pts, pts_raw = pip.transform(tmp, preprocessing=True)
        assert(tmp.shape==out.shape)
        assert(np.all(out==0))
        assert(pts==None)

    def test_preprocessing(self):
        pip = FACE_pipeline(
                histogram_normalization=True,
                grayscale=True
                )
        out, pts, pts_raw = pip.transform(img, preprocessing=True)
        assert(img.shape[:2]==out.shape[:2])
        assert(out.shape[-1]==1)
        assert(out.max()<=255)
        assert(out.min()>=0)
        assert(pts==None)

    def test_augmentation(self):
        pip = FACE_pipeline(
                histogram_normalization=True,
                grayscale=True,
                rotation_range = 10,
                width_shift_range = 0.05,
                height_shift_range = 0.05,
                zoom_range = 0.05,
                fill_mode = 'edge',
                random_flip = True,
                )


        out, pts, pts_raw = pip.transform(
                img, 
                preprocessing=False,
                augmentation=True
                )
        assert(pts==None)
        assert(img.shape==out.shape)


        out, pts, pts_raw = pip.transform(
                img, 
                preprocessing=True,
                augmentation=True
                )
        assert(pts==None)
        assert(out.shape[-1]==1)
        assert(img.shape[:2]==out.shape[:2])

    def test_face_detection(self):
        pip = FACE_pipeline(
                output_size = [160,240],
                face_size = 160,
                )
        out, pts, pts_raw = pip.transform(img, face_detect=True)
        assert(out.shape==(240,160,3))
        assert(pts.shape==(68,2))
        assert(pts_raw.shape==(68,2))

    def test_full_pipeline(self):
        pip = FACE_pipeline(
                output_size = [160,240],
                face_size = 160,
                histogram_normalization=True,
                grayscale=True,
                rotation_range = 10,
                width_shift_range = 0.05,
                height_shift_range = 0.05,
                zoom_range = 0.05,
                gaussian_range = 2,
                fill_mode = 'edge',
                random_flip = True,
                )
        out, pts, pts_raw = pip.transform(
                img, 
                face_detect=True, 
                preprocessing=True, 
                augmentation=True
                )
        assert(out.shape==(240,160,1))
        assert(pts.shape==(68,2))

    def test_full_pipeline_variations(self):
        img = imread(pwd+'/data/images/test_08.jpg')
        pip = FACE_pipeline(
                output_size = [256,256],
                face_size = 224,
                histogram_normalization=True,
                grayscale=False,
                rotation_range = 15.,
                width_shift_range = 0.05,
                height_shift_range = 0.05,
                zoom_range = 0.05,
                gaussian_range = 5,
                fill_mode = 'edge',
                random_flip = True,
                )

        out_y= []
        for x in range(3):
            out_x = []
            for y in range(2):
                out, pts, pts_raw = pip.transform(
                        img, 
                        face_detect=True, 
                        preprocessing=True, 
                        augmentation=True
                        )
                assert(out.shape==(256,256,3))
                assert(pts.shape==(68,2))
                out_x.append(out)

            out_y.append(np.vstack(out_x))
        out_grid = np.hstack(out_y)
        out_grid = out_grid-out_grid.min()
        out_grid = out_grid/out_grid.max()
        imsave('train.jpg',np.uint8(out_grid))

        out, pts, pts_raw = pip.transform(
                img, 
                face_detect=True, 
                preprocessing=True, 
                augmentation=False,
                )
        assert(out.shape==(256,256,3))
        assert(pts.shape==(68,2))
        imsave('test.jpg',np.uint8(out))





if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__file__, env={'NOSE_NOCAPTURE' : 1})
