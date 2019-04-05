from __future__ import print_function

import keras
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta

from mnist import MNIST
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import textwrap
import gzip
import os
from os.path import join
import sys
import time
from struct import unpack
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#print(K.tensorflow_backend._get_available_gpus())

image_dir = os.getcwd()+"\\gzip"

# Gets the Specified Dataset
def classes(x):
    return {
        1: ('balanced', 47, 112800),
        2: ('bymerge', 47, 697932),
        3: ('byclass', 62, 697932),
        4: ('letters', 26, 124800),
        5: ('digits', 10, 240000),
        6: ('mnist', 10, 60000),
    }[x]

# IDX file format can be found at the bottom of http://yann.lecun.com/exdb/mnist/
def IDX(filename):
    with gzip.open(filename, 'rb') as f:
        z, dtype, dim = unpack('>HBB', f.read(4))
        shape = tuple(unpack('>I', f.read(4))[0] for d in range(dim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def load_emnist_type1(image_dir, dataset):
    train_X = IDX(join(image_dir, 'emnist-'+dataset+'-train-images-idx3-ubyte.gz'))
    train_y = IDX(join(image_dir, 'emnist-'+dataset+'-train-labels-idx1-ubyte.gz'))
    test_X = IDX(join(image_dir, 'emnist-'+dataset+'-test-images-idx3-ubyte.gz'))
    test_y = IDX(join(image_dir, 'emnist-'+dataset+'-test-labels-idx1-ubyte.gz'))

    return train_X, train_y, test_X, test_y

'''
# Testing a different way to load the dataset
def load_emnist_type2(image_dir, dataset, xy_shape):
	emnist_data = MNIST(path=image_dir+'\\', return_type='numpy')
	emnist_data.select_emnist(dataset)
	X, y = emnist_data.load_training()
	# print(X.shape, y.shape)

	X = X.reshape(xy_shape, 28, 28)
	y = y.reshape(xy_shape, 1)
	# print(X.shape, y.shape)

	y = y-1
	# print(X.shape, y.shape)

	return train_test_split(X, y, test_size=0.25, random_state=111)
'''

# Setting up the CNN Structure
def createModel(num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Comple the model
    model.compile(loss=categorical_crossentropy, optimizer=Adadelta(), metrics=['accuracy'])
    return model


def main():

    if not os.path.exists('gzip\\'):
        print("Please Download 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip' file containing the Datasets and unzip into this directory.")
        return

    print(textwrap.dedent("""\
        Datasets to choose from:
            1 = balanced
            2 = bymerge
            3 = byclass
            4 = letters
            5 = digits
            6 = mnist
    	"""))

    whichset = int(input("What dataset would you like build a model of?, choose 1 - 6 [1]: ") or "1")
    if whichset < 1 and whichset > 6:
        print("Need to input one of the specified integers 1 through 6")
        return

    choosen = classes(whichset)
    dataset = choosen[0]
    num_classes = choosen[1]
    xy_shape = choosen[2]
    #print("dataset = {0}, number of classes = {1}, xy-shape = {2}".format(dataset,str(num_classes),str(xy_shape)))

    batch_size = int(input("Batch Size [512]: ") or "512")

    epochs = int(input("Epoch [13]: ") or "13")

    name = input("Model File's Name [{0}]: ".format(dataset)) or dataset
    model_path = os.getcwd()+'\\'+name+'.h5'
    #print("Model path/name = {0}".format(model_path))
    #method = int(input("Choose how to load the data from the dataset 1 or 2 [1]: ") or "1")

    raw_train_X, raw_train_y, raw_test_X, raw_test_y = load_emnist_type1(image_dir, dataset) #if (method == 1) else load_emnist_type2(image_dir, dataset, xy_shape)

    #____________________________________________Data Processing And Model Training Begins_______________________________________________
    # Rescale the Images by Dividing Every Pixel in Every Image by 255
    train_X = raw_train_X.reshape(raw_train_X.shape[0], 28, 28, 1).astype('float32')/255
    test_X = raw_test_X.reshape(raw_test_X.shape[0], 28, 28, 1).astype('float32')/255

    # Convert class vectors to binary class matrices
    train_y = to_categorical(raw_train_y) #, num_classes)
    test_y = to_categorical(raw_test_y) #, num_classes)
 
    model = createModel(num_classes)

    # Train the model
    t1 = time.time()
    fit = model.fit(train_X, train_y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(test_X, test_y))
    t2 = time.time()
    print('Elapsed time: %ds' % (t2 - t1))

    # Save the model
    model.save(model_path)

    # Evaluate  model for Accuracy and Loss
    print(model.layers)
    results = model.evaluate(test_X, test_y)
    print('Loss: %.2f%%, Accuracy: %.2f%%' % (results[0]*100, results[1]*100))

    # Display graph for Model's Train Stats
    plt.figure(figsize=(12, 6), dpi=96)
    plt.subplot(1, 2, 1)
    plt.plot(fit.history['loss'])
    plt.plot(fit.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train', 'test'], loc='upper left')
    plt.subplot(1, 2, 2)
    plt.plot(fit.history['acc'])
    plt.plot(fit.history['val_acc'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'test'], loc='upper left')
    plt.tight_layout()
    plt.show()


main()