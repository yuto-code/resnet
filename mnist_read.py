from __future__ import print_function
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.models import model_from_json
from PIL import Image
from skimage.io import imread, imshow, imread_collection, concatenate_images
import numpy as np
import matplotlib.pyplot as plt


def main():
    """
    学習したモデルを読み込み数字を当てる
    """
    # モデルの読み込み
    model = model_from_json(open('mnist.json', 'r').read())
    # 重みの読み込み
    model.load_weights('mnist.h5')
    
    im = np.array(Image.open('1.png'))
    imshow(im)
    img = np.zeros([1,28,28,1])
    for i in range(28):
        for j in range(28):
            img[0][i][j][0] = im[i][j][0]/255
    
    y=model.predict(img)
    print(max(y[0]))
    for i in range(10):
        if y[0][i]==max(y[0]):
            print('この数字は')
            print(i)
            print('です')

if __name__ == '__main__':
    main()