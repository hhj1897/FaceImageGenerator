import EmoData 

path_to_shape_model = '/homes/rw2614/projects/shape_model/shape_predictor_68_face_landmarks.dat'

fg = EmoData.provider.Facial_Expressions(
    histogram_normalization = True,
    make_grayscale = True,
    path_to_shape_model = path_to_shape_model,
    output_size = [128,192],
    face_size = 128,
    rotation_range = 25,
    width_shift_range = 0.05,
    height_shift_range = 0.05,
    zoom_range = 0.05,
    )


# load images
from skimage.io import imread
img = imread('./test_data/images/test_01.jpg')
print(img.shape)

# pipeline 1) image goes trough the pipeline without any processings
out, pts = fg.run_pipeline(img, extract_bbox=False, preprocessing=False, augment=False)
print(out.shape, out.max(), out.min())

# pipeline 2) face detection, cropping and alignment 
out, pts = fg.run_pipeline(img, extract_bbox=True,  preprocessing=False, augment=False)
print(out.shape, out.max(), out.min())

# pipeline 3) face detection, cropping, alignment and normalization
out, pts = fg.run_pipeline(img, extract_bbox=True,  preprocessing=True,  augment=False)
print(out.shape, out.max(), out.min())

# pipeline 3) face detection, cropping, alignment, normalization and augmentation
out, pts = fg.run_pipeline(img, extract_bbox=True,  preprocessing=True,  augment=False)
print(out.shape, out.max(), out.min())
