import numpy as np
from skimage import transform

def rotate(img, deg):
    '''
    '''
    # Shift the image so that the centre is around the origin
    center = np.array([int(np.floor(i)/2) for i in img.shape[:2][::-1]])
    trans = transform.SimilarityTransform(translation=-center)

    # Rotate 
    trans += transform.SimilarityTransform(rotation=np.pi/180*deg)

    # Shift the image back
    trans += transform.SimilarityTransform(translation=center)
    return trans 

def zoom(img, scale):
    '''
    '''
    # Shift the image so that the centre is around the origin
    center = np.array([int(np.floor(i)/2) for i in img.shape[:2][::-1]])
    trans = transform.SimilarityTransform(translation=-center)

    # Scale 
    trans += transform.SimilarityTransform(scale=1/(scale+1))

    # Shift the image back
    trans += transform.SimilarityTransform(translation=center)
    return trans 

def width_shift(img, shift):
    '''
    '''
    return transform.SimilarityTransform(translation=[shift*img.shape[1],0])

def height_shift(img, shift):
    '''
    '''
    return transform.SimilarityTransform(translation=[0,shift*img.shape[0]])
