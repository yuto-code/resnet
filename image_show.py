
from __future__ import print_function
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, imread_collection, concatenate_images
import numpy as np
from PIL import Image
import numpy as np


batch_size = 32
nb_classes = 10

nb_epoch = 200
data_augmentation = True

img_rows, img_cols = 28, 28
img_channels = 1

(X_train, y_train), (X_test, y_test) = mnist.load_data()

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
"""
K=4
imshow(X_train[K])
plt.show()
for i in range(10):
    if Y_train[K][i]==1:
        print(i)
"""
im = np.array(Image.open('1.jpg'))
imshow(im)