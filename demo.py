import emotion_data

fg = emotion_data.provider.Facial_Expressions(
            histogram_normalization=True,
            mean_std_normalization=True,
            allignment_type = False,
            downscaling = 1,
            rotation_range = 10,
            width_shift_range = 0.05,
            height_shift_range = 0.05,
            zoom_range = 0.05,
            fill_mode = 'edge',
            random_flip = True,
            path_to_shape_model = None,
        )


img_tr = fg.flow_from_hdf5('./test_data/test_data_64_96_1.h5','img', batch_size=16, augment=False)

# get 10 batches
for i in range(10):
    img = next(img_tr)
    print(img.shape)


# apply preprocessing on single frame
# out = fg.face_input_frame
