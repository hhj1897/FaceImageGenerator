import emotion_data


fg = emotion_data.provider.Facial_Expressions(
        histogram_normalization=True,
        mean_std_normalization=True,
        greyscale=True,
        rotation_range = 10,
        width_shift_range = 0.05,
        image_size = [128, 192],
        face_size = 128,
        height_shift_range = 0.05,
        zoom_range = 0.05,
        fill_mode = 'edge',
        random_flip = True,
        path_to_shape_model = '/homes/rw2614/projects/shape_model',
        allignment_type = 'similarity',
        )


img_tr_folder = fg.flow_from_folder(
        './test_data/images','jpg', 
        batch_size=3, 
        extract_bbox=True, 
        preprocess=True, 
        augment=False, 
        # add_mask=True, 
        save_to_dir='./tmp/'
        )

for i in range(5):
    next(img_tr_folder)
