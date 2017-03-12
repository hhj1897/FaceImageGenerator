from EmoData.image_pipeline import FACE_pipeline 
import os
from skimage.io import imread
import glob
import numpy as np
pwd = os.path.dirname(os.path.abspath(__file__))

batch = []
for f in glob.glob(pwd+'/data/images/test_*.jpg'):
    batch.append(imread(f))

pip = FACE_pipeline(
        output_size = [160,240],
        face_size = 160,
        histogram_normalization=True,
        grayscale=True,
        rotation_range = 10,
        width_shift_range = 0.05,
        height_shift_range = 0.05,
        zoom_range = 0.05,
        fill_mode = 'edge',
        random_flip = True,
        )

class testcase:

    def _test_batch_transform(self):

        out, pts = pip.batch_transform(batch, True, False)
        assert(out.shape==(12, 240, 160, 3))
        assert(pts.shape==(12, 68, 2))

        out, pts = pip.batch_transform(batch, True, True)
        assert(out.shape==(12, 240, 160, 1))
        assert(pts.shape==(12, 68, 2))

    def _test_small_batch(self):

        sub_batch = batch[:2]
        out, pts = pip.batch_transform(sub_batch, True, False)
        assert(out.shape==(2, 240, 160, 3))
        assert(pts.shape==(2, 68, 2))

        sub_batch = batch[:1]
        out, pts = pip.batch_transform(sub_batch, True, False)
        assert(out.shape==(1, 240, 160, 3))
        assert(pts.shape==(1, 68, 2))

    def test_partially_annotated(self):

        sub_batch = batch
        img, _ = pip.batch_transform(sub_batch, True)
        # img[:4] = -np.ones_like(img[:4])
        img = -np.ones_like(img)
        img, _ = pip.batch_transform(img, preprocessing=True)



if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__file__, env={'NOSE_NOCAPTURE' : 1})
