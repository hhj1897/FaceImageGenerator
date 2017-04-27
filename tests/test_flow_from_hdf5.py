from FaceImageGenerator.provider import flow_from_hdf5
import os
import numpy as np
pwd = os.path.dirname(os.path.abspath(__file__))

def _data_iterator(gen_dict):
    while True:
        img = next(gen_dict['img'])
        pts = next(gen_dict['pts'])
        yield img, pts

class testcase:
    def test_read_file(self):
        gen_dict = flow_from_hdf5(pwd+'/data/test_data_64_96_1.h5', batch_size=15)
        assert(gen_dict['nb_samples']==100)
        assert(gen_dict['nb_batches']==7)

    def test_padding(self):
        gen_dict = flow_from_hdf5(pwd+'/data/test_data_64_96_1.h5', batch_size=15, padding=-1)
        gen = _data_iterator(gen_dict)
        assert(gen_dict['nb_samples']==100)
        assert(gen_dict['nb_batches']==7)

        out_img, out_pts = [], []
        for i in range(gen_dict['nb_batches']):
            img, pts = next(gen)
            out_img.append(img)
            out_pts.append(pts)

        out_img = np.vstack(out_img)
        out_pts = np.vstack(out_pts)

        print(out_img.shape)

        assert(out_img.shape[0]==105)
        assert(out_pts.shape[0]==105)

    def test_iterate_trough_file(self):
        gen_dict = flow_from_hdf5(pwd+'/data/test_data_64_96_1.h5', batch_size=15)
        gen = _data_iterator(gen_dict)

        out_img, out_pts = [], []
        for i in range(gen_dict['nb_batches']):
            img, pts = next(gen)
            out_img.append(img)
            out_pts.append(pts)

        out_img = np.vstack(out_img)
        out_pts = np.vstack(out_pts)

        assert(out_img.shape[0]==100)
        assert(out_pts.shape[0]==100)





if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__file__, env={'NOSE_NOCAPTURE' : 1})
