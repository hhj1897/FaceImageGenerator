import EmoData 
import numpy as np

path_to_shape_model = '/homes/rw2614/projects/shape_model/shape_predictor_68_face_landmarks.dat' 



################################################################
# landmark detection, alignment, normalization and  augmentation
################################################################
fg = EmoData.provider.Facial_Expressions(
    histogram_normalization = True,
    mean_std_normalization = True,
    make_grayscale = True,
    path_to_shape_model = path_to_shape_model,
    rotation_range = 10,
    width_shift_range = 0.05,
    height_shift_range = 0.05,
    zoom_range = 0.05,
    )

img_reader = fg.flow_from_hdf5( 
        path_to_file = './test_data/test_data_64_96_1.h5',
        group_name = 'img', 
        batch_size = 10,
        preprocessing = True,
        augment = True,
        save_to_dir = './tmp/'
        )

def to_class_vector(one_hot_matrix):
    return np.argmax(one_hot_matrix,1)
lab_reader = fg.flow_from_hdf5( 
        path_to_file = './test_data/test_data_64_96_1.h5',
        postprocessing = to_class_vector,
        group_name = 'lab', 
        )

pts_reader = fg.flow_from_hdf5( 
        path_to_file = './test_data/test_data_64_96_1.h5',
        group_name = 'pts', 
        )

data_reader = zip(img_reader, pts_reader, lab_reader)


img, pts, lab = next(data_reader)
