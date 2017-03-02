from EmoData.image_pipeline import FACE_pipeline 
import os
from skimage.io import imread
import glob
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

    def test_batch_transform(self):

        out, pts = pip.batch_transform(batch, face_detect=True)
        assert(out.shape==(12, 240, 160, 3))
        assert(pts.shape==(12, 68, 2))



if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__file__, env={'NOSE_NOCAPTURE' : 1})
