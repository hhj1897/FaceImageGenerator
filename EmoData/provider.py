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
