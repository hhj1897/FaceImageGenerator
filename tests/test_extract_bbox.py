import EmoData 
path_to_shape_model = '/homes/rw2614/projects/shape_model/shape_predictor_68_face_landmarks.dat' 

###############################################################
# reading images from folder
###############################################################
fg = EmoData.provider.Facial_Expressions()

img_reader = fg.flow_from_folder( 
        path_to_folder = './test_data/images',
        suffix = 'jpg', 
        batch_size = 10,
        # save_to_dir = './tmp/'
        )

batch_img, _ = next(img_reader)

print(len(batch_img))
print(batch_img[0].shape)


################################################################
# landmark detection and alignment
################################################################
fg = EmoData.provider.Facial_Expressions(
    path_to_shape_model = path_to_shape_model
    )

img_reader = fg.flow_from_folder( 
        path_to_folder = './test_data/images',
        suffix = 'jpg', 
        batch_size = 10,
        extract_bbox = True,
        # save_to_dir = './tmp/'
        )

batch_img, batch_pts = next(img_reader)

print(len(batch_img))
print(batch_img[0].shape)
print(len(batch_pts))
print(batch_pts[0].shape)


################################################################
# histogram_normalization and mean_std_normalization 
################################################################
fg = EmoData.provider.Facial_Expressions(
    histogram_normalization = True,
    mean_std_normalization = True,
    make_grayscale = True,
    )

img_reader = fg.flow_from_folder( 
        path_to_folder = './test_data/images',
        suffix = 'jpg', 
        batch_size = 10,
        preprocessing = True,
        save_to_dir = './tmp/'
        )

batch_img, _ = next(img_reader)

print(len(batch_img))
print(batch_img[0].shape)

###############################################################
# landmark detection, alignment and normalization
###############################################################
fg = EmoData.provider.Facial_Expressions(
    histogram_normalization = True,
    mean_std_normalization = True,
    make_grayscale = True,
    path_to_shape_model = path_to_shape_model
    )

img_reader = fg.flow_from_folder( 
        path_to_folder = './test_data/images',
        suffix = 'jpg', 
        batch_size = 10,
        extract_bbox = True,
        preprocessing = True,
        save_to_dir = './tmp/'
        )

batch_img, batch_pts = next(img_reader)

print(len(batch_img))
print(batch_img[0].shape)
print(len(batch_pts))
print(batch_pts[0].shape)


################################################################
# landmark detection, alignment, normalization and  augmentation
################################################################
fg = EmoData.provider.Facial_Expressions(
    histogram_normalization = True,
    mean_std_normalization = True,
    make_grayscale = True,
    path_to_shape_model = path_to_shape_model,
    output_size = [128,192],
    face_size = 128,
    rotation_range = 90,
    width_shift_range = 0.05,
    height_shift_range = 0.05,
    zoom_range = 0.05,
    )

img_reader = fg.flow_from_folder( 
        path_to_folder = './test_data/images',
        suffix = 'jpg', 
        batch_size = 10,
        extract_bbox = True,
        preprocessing = True,
        augment = True,
        save_to_dir = './tmp/'
        )

batch_img, batch_pts = next(img_reader)

print(len(batch_img))
print(batch_img[0].shape)
print(len(batch_pts))
print(batch_pts[0].shape)
