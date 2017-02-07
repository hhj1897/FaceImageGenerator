import EmoData 

################################################################
# landmark detection, alignment, normalization and  augmentation
################################################################
fg = EmoData.provider.Facial_Expressions(
    histogram_normalization = True,
    mean_std_normalization = True,
    make_grayscale = True,
    rotation_range = 10,
    width_shift_range = 0.05,
    height_shift_range = 0.05,
    zoom_range = 0.05,
    )


gen = fg.flow_from_hdf5_list( 
        list_of_paths = ['./test_data/test_data_64_96_1.h5'] * 3 ,
        batch_size = 10,
        downscaling = 2,
        downsampling = 5,
        shuffle = True,
        augment = True,
        preprocessing = True,
        )

test = next(gen['img'])
assert(test.shape==(10,48,32,1))

test = next(gen['lab'])
assert(test.shape==(10,12,6))

test = next(gen['pts'])
assert(test.shape==(10,2,68))
