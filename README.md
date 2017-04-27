# FaceImageGenerator


A pipeline for automatic facial feature extraction and image augmetnation.
This project is heavly inspired by the ImageDataGenerator from [keras](https://keras.io/preprocessing/image/)
If you have images with faces and want to train a CNN, you might find something useful here.

##### Install instructions:
requires:
* Python (>= 2.6 or >= 3.3),
* skimage (>= 0.12.3),
* h5py (>= 2.6.0),
* Numpy(>= 1.12.1),
* dlib 

First, get the code from Github:
```sh
git clone git@github.com:RWalecki/FaceImageGenerator.git
```

Next, go into the directory where the clone was placed and run the installation script:
```
cd FaceImageGenerator
python setup.py install
```

##### Test the installation:
Once the installation is done, you should run the nosetests before using the project.
Therefore, run:
```
nosetests .
```
The tests should not take longer than a few seconds. You are ready to use FaceImageGenerator. Enjoy!
