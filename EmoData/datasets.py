import numpy as np
import glob
from .provider import flow_from_hdf5, flow_from_np_array


def fera2015(batch_size=32):
    '''
    '''
    TR = flow_from_hdf5('/homes/rw2614/data/NEW/similarity_160_240/fera2015_tr.h5', batch_size)
    TE = flow_from_hdf5('/homes/rw2614/data/NEW/similarity_160_240/fera2015_te.h5', batch_size)
    return TR, TE 

def disfa(batch_size=32):
    '''
    '''
    TR = flow_from_hdf5('/homes/rw2614/data/NEW/similarity_160_240/disfa_tr.h5', batch_size)
    TE = flow_from_hdf5('/homes/rw2614/data/NEW/similarity_160_240/disfa_te.h5', batch_size)
    return TR, TE

from keras.datasets import mnist as _mnist
def mnist(batch_size=32):
    (X_train, y_train), (X_test, y_test) = _mnist.load_data()
    TR = flow_from_np_array(X_train, y_train, batch_size)
    TE = flow_from_np_array(X_test, y_test, batch_size)
    return TR, TE
