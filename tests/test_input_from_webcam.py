# from skimage.io import imread
import emotion_data
path_to_shape_model = '/homes/rw2614/projects/shape_model/shape_predictor_68_face_landmarks.dat'

fg = emotion_data.provider.Facial_Expressions(
    histogram_normalization = True,
    mean_std_normalization = True,
    make_grayscale = True,
    add_mask = True,
    path_to_shape_model = path_to_shape_model,
    output_size = [240,360],
    face_size = 180,
    rotation_range = 15,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    zoom_range = 0.1,
    )


import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    print(frame.shape)

    frame , pts = fg.run_pipeline(frame[::2,::2], extract_bbox=True,  preprocessing=True,  augment=False)

    frame-=frame.min()
    frame/=frame.max()
    frame=np.uint8(frame*255)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
