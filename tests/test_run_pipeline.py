from skimage.io import imread
import EmoData 
path_to_shape_model = '/homes/rw2614/projects/shape_model/shape_predictor_68_face_landmarks.dat'

img = imread('./test_data/images/test_01.jpg')
print(img.shape)

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

out, pts = fg.run_pipeline(img, extract_bbox=False, preprocessing=False, augment=False)
print(out.shape, out.max(), out.min())

out, pts = fg.run_pipeline(img, extract_bbox=True,  preprocessing=False, augment=False)
print(out.shape, out.max(), out.min())

out, pts = fg.run_pipeline(img, extract_bbox=True,  preprocessing=True,  augment=False)
print(out.shape, out.max(), out.min())

for i in range(10):
    out, pts = fg.run_pipeline(img[::3,::3], extract_bbox=True,  preprocessing=True,  augment=False)
    print(i, out.shape, out.max(), out.min())
