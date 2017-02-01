import emotion_data
from skimage.io import imread

test=0
for histogram_normalization in [True,False]:
    for mean_std_normalization in [True,False]:
        for greyscale in [True,False]:
            for preprocessing in [True,False]:
                for augment in [True,False]:

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
                            extract_bbox=True, 
                            preprocess=preprocessing, 
                            augment=augment, 
                            add_mask=True, 
                            save_to_dir='./tmp/'
                            )

                    batch = next(img_tr_folder)

                    out_stored = imread('./tmp/00001_00001.jpg')
                    out_from_batch = batch[0]


                    test+=1
                    print(test)
                    print(histogram_normalization)
                    print(mean_std_normalization)
                    print(greyscale)
                    print(preprocessing)
                    print(augment)
                    assert(out_stored.shape[:2] == out_from_batch.shape[:2])
                    assert(out_stored.mean()>0)
