import numpy as np
import glob
from .provider import flow_from_hdf5


def fera2015(batch_size=32):
    '''
    '''
    TR = flow_from_hdf5('/homes/rw2614/data/NEW/similarity_160_240/fera2015_tr.h5', batch_size)
    TE = flow_from_hdf5('/homes/rw2614/data/NEW/similarity_160_240/fera2015_te.h5', batch_size)
    return TR, TE 

def disfa(batch_size=32):
    '''
    '''
    TR = flow_from_hdf5('/homes/rw2614/data/NEW/similarity_160_240/disfa_te.h5', batch_size)
    TE = flow_from_hdf5('/homes/rw2614/data/NEW/similarity_160_240/disfa_tr.h5', batch_size)
    return TR, TE
