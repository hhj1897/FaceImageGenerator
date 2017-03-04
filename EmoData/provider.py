import h5py
import math

def flow_from_hdf5(
        path_to_file, 
        batch_size=64,
        ):
    '''
    '''
    f = h5py.File(path_to_file)
    
    # get the sice of the first group in the hdf5 file
    data = f[[i for i in f.keys()][0]]

    nb_samples = data.shape[0]
    nb_batches = math.ceil(nb_samples/batch_size)

    def _make_generator(data):

        t0, t1  = 0, batch_size

        while True:

            t1 = min( nb_samples, t1 )
            if t0 >= nb_samples:
                t0, t1 = 0, batch_size

            batch = data[t0:t1]

            t0 += batch_size
            t1 += batch_size

            yield batch

    res_gen = {}
    res_gen['nb_samples']=nb_samples
    res_gen['nb_batches']=nb_batches
    for key in f:
        res_gen[key] = _make_generator(f[key])
    return res_gen

# import h5py
# import math
# import threading


# def _make_h5_generator(group, batch_size):

    # nb_samples = group.shape[0]
    # t0, t1  = 0, batch_size

    # while True:

        # t1 = min( nb_samples, t1 )
        # if t0 >= nb_samples:
            # t0, t1 = 0, batch_size

        # batch = group[t0:t1]

        # t0 += batch_size
        # t1 += batch_size

        # yield batch


# class flow_from_hdf5():
    # '''
    # '''
    # def __init__(self, path_to_file, batch_size=16, verbose=0):
        # self.path_to_file = path_to_file
        # self.batch_size = batch_size
        # self.lock = threading.Lock()

        # # get the size of the first group in the hdf5 file
        # # to compute number of sample and batches
        # f = h5py.File(path_to_file)
        # data = f[[i for i in f.keys()][0]]
        # self.nb_samples = data.shape[0]
        # self.nb_batches = math.ceil(self.nb_samples/batch_size)
        # self.groups = sorted(list(f.keys()))

        # if verbose:
            # print('path_to_file:'.ljust(12), self.path_to_file)
            # print('groups:'.ljust(12), self.groups)
            # print('batch_size:'.ljust(12), self.batch_size)
            # print('nb_samples:'.ljust(12), self.nb_samples)
            # print('nb_batches:'.ljust(12), self.nb_batches)


        # self.generator = {}
        # for group in self.groups:
            # group_gen = _make_h5_generator(f[group], batch_size)
            # self.generator[group] = group_gen

    # def next(self, group):
        # with self.lock:
            # return next(self.generator[group])
