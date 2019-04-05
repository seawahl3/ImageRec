import sys
import os
import keras
import cv2
import numpy as np
from pytesseract import image_to_string
from keras.models import load_model
from spellchecker import SpellChecker

# Global Variables
wordChecker = SpellChecker()

"""
# For Models based on the 'byclass' dataset
characters = {
 1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8',
 10: '9', 11: 'A', 12: 'B', 13: 'C', 14: 'D', 15: 'E', 16: 'F', 17: 'G',
 18: 'H', 19: 'I', 20: 'J', 21: 'K', 22: 'L', 23: 'M', 24: 'N', 25: 'O',
 26: 'P', 27: 'Q', 28: 'R', 29: 'S', 30: 'T', 31: 'U', 32: 'V', 33: 'W',
 34: 'X', 35: 'Y', 36: 'Z', 37: 'a', 38: 'b', 39: 'c', 40: 'd', 41: 'e',
 42: 'f', 43: 'g', 44: 'h', 45: 'i', 46: 'j', 47: 'k', 48: 'l', 49: 'm',
 50: 'n', 51: 'o', 52: 'p', 53: 'q', 54: 'r', 55: 's', 56: 't', 57: 'u',
 58: 'v', 59: 'w', 60: 'x', 61: 'y', 62: 'z'}
"""

# For Models based on the 'balanced' or 'bymerge' datasets
characters = {
 1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8',
 10: '9', 11: 'A', 12: 'B', 13: 'C', 14: 'D', 15: 'E', 16: 'F', 17: 'G',
 18: 'H', 19: 'I', 20: 'J', 21: 'K', 22: 'L', 23: 'M', 24: 'N', 25: 'O',
 26: 'P', 27: 'Q', 28: 'R', 29: 'S', 30: 'T', 31: 'U', 32: 'V', 33: 'W',
 34: 'X', 35: 'Y', 36: 'Z', 37: 'a', 38: 'b', 39: 'd', 40: 'e', 41: 'f',
 42: 'g', 43: 'h', 44: 'n', 45: 'q', 46: 'r', 47: 't'}


"""
# For Models based on the 'letters' dataset
characters = {
 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h',
 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p',
 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x',
 25: 'y', 26: 'z'}
"""

"""
# For Models based on the 'digits' or 'mnist' datasets
characters = {
 1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8',
 10: '9'}
"""

def findWord(folderpath):
    # Empty string to add each of the letters to as they are found
    word = ""
     for image in enumerate(os.listdir(folderpath)):
         word += letterIdentifier(folderpath + '/' + image[1])

    # return letterIdentifier(folderpath)
    return wordChecker.correction(word)
    # return word


def letterIdentifier(imageName):
    # Load the models built in the previous steps
    keras.backend.clear_session()
    cnn_model = load_model('balanced.h5')
    prediction = len(characters)
    print(prediction)
    # Load and convert the image it to grayscale
    image = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
    # Resize the image to model dimensions and convert it to a numpy array
    image = np.array(cv2.resize(image, (28, 28), 1)).astype('float32') / 255
    # Define array of probabilities
    prediction = cnn_model.predict(image.reshape(1, 28, 28, 1))[0]
    # Index of the highest probablility
    predictionIn = np.argmax(prediction)
    #print(prediction[predictionIn])
    #print(predictionIn)
    prob = int(prediction[predictionIn] * 100)
    letter = str(characters[int(predictionIn) + 1]) if prob > 75 else image_to_string(imageName, config='--psm 10 --dpi 70')
    return letter
