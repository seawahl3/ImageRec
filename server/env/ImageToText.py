import shutil
import sys
import os
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr
import cv2
import numpy as np
from keras.models import load_model
from spellchecker import SpellChecker

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
wordChecker = SpellChecker()

# Global Variables


characters = {
 1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8',
 10: '9', 11: 'A', 12: 'B', 13: 'C', 14: 'D', 15: 'E', 16: 'F', 17: 'G',
 18: 'H', 19: 'I', 20: 'J', 21: 'K', 22: 'L', 23: 'M', 24: 'N', 25: 'O',
 26: 'P', 27: 'Q', 28: 'R', 29: 'S', 30: 'T', 31: 'U', 32: 'V', 33: 'W',
 34: 'X', 35: 'Y', 36: 'Z', 37: 'a', 38: 'b', 39: 'd', 40: 'e', 41: 'f',
 42: 'g', 43: 'h', 44: 'n', 45: 'q', 46: 'r', 47: 't', 48: '-'}
"""
characters = {
 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h',
 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p',
 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x',
 25: 'y', 26: 'z', 27: '-'}
"""

# Load the models built in the previous steps

# print(os.getcwd())
cnn_model = load_model('temp.h5')


def findWord(folderpath):
    # Empty string to add each of the letters to as they are found
    word = ""
    for image in enumerate(os.listdir(folderpath)):
        print(folderpath + '/' + image)
        word += letterIdentifier(folderpath + '/' + image)
    # Delete the temp image directory
    # shutil.rmtree(lettersDirectory)
    # return wordChecker.correction(word)
    return word


def letterIdentifier(imageName):
    prediction = 47
    # Load the image and convert it to grayscale
    image = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
    # Resize the image to model input dimension
    image = cv2.resize(image, (28, 28), 1)
    # Convert img to a numpy array
    image = np.array(image)
    # Preprocess the image
    # (similar preprocessing was performed before training)
    image = image.astype('float32') / 255
    # Define prediction variables
    prediction = cnn_model.predict(image.reshape(1, 28, 28, 1))[0]
    print(prediction)
    prediction = np.argmax(prediction)
    # print(prediction)
    return str(characters[int(prediction) + 1])

