import emotion_data

test=0
for histogram_normalization in [True,False]:
    for mean_std_normalization in [True,False]:
        for greyscale in [True,False]:
            for preprocessing in [True,False]:
                for augment in [True,False]:
                    for extract_bbox in [True,False]:

                        fg = emotion_data.provider.Facial_Expressions(
                                    histogram_normalization=histogram_normalization,
                                    mean_std_normalization=mean_std_normalization,
                                greyscale=greyscale,
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
                                extract_bbox=extract_bbox, 
                                preprocess=preprocessing, 
                                augment=augment, 
                                add_mask=True, 
                                )

                        batch = next(img_tr_folder)

                        if extract_bbox:
                            assert(len(batch)==2)
                            assert(batch[0].shape[0]==3)
                            assert(batch[1].shape[0]==3)
                        else:
                            assert(len(batch)==3)

                        test+=1
                        print(test)
