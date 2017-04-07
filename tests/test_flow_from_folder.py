from EmoData.provider import flow_from_folder
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
        gen_dict = flow_from_folder(pwd+'/data/images', batch_size=2)
        assert(gen_dict['nb_samples']==12)
        assert(gen_dict['nb_batches']==6)

    def test_padding(self):
        gen_dict = flow_from_folder(pwd+'/data/images', batch_size=7, padding=-1)

        assert(gen_dict['nb_samples']==12)
        assert(gen_dict['nb_batches']==2)
        X0 = next(gen_dict['img'])
        X1 = next(gen_dict['img'])
        assert(np.all(X1[-1]==-1))

        X0 = next(gen_dict['lab'])
        X1 = next(gen_dict['lab'])
        assert(np.all(X1[-1]==-1))



if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__file__, env={'NOSE_NOCAPTURE' : 1})
