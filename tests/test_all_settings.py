import EmoData 

test=0
for histogram_normalization in [True,False]:
    for mean_std_normalization in [True,False]:
        for grayscale in [True,False]:
            for preprocessing in [True,False]:
                for augment in [True,False]:

                    fg = EmoData.provider.Facial_Expressions(
                                histogram_normalization=histogram_normalization,
                                mean_std_normalization=mean_std_normalization,
                            make_grayscale=grayscale,
                            rotation_range = 10,
                            width_shift_range = 0.05,
                            output_size = [128, 192],
                            face_size = 128,
                            height_shift_range = 0.05,
                            zoom_range = 0.05,
                            fill_mode = 'edge',
                            random_flip = True,
                        )


                    img_tr_folder = fg.flow_from_folder(
                            './test_data/images','jpg', 
                            batch_size=3, 
                            preprocessing=preprocessing, 
                            augment=augment, 
                            )

                    batch = next(img_tr_folder)

                    test+=1
                    print(test)
