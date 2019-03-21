from keras.models import load_model
from spellchecker import SpellChecker
import shutil
import os 
import numpy as np
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Global Variables

characters = {
    1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8', 10: '9',
    11: 'A', 12: 'B', 13: 'C', 14: 'D', 15: 'E', 16: 'F', 17: 'G', 18: 'H', 19: 'I', 20: 'J',
    21: 'K', 22: 'L', 23: 'M', 24: 'N', 25: 'O', 26: 'P', 27: 'Q', 28: 'R', 29: 'S', 30: 'T',
    31: 'U', 32: 'V', 33: 'W', 34: 'X', 35: 'Y', 36: 'Z', 37: 'a', 38: 'b', 39: 'c', 40: 'd',
    41: 'e', 42: 'f', 43: 'g', 44: 'h', 45: 'i', 46: 'j', 47: 'k', 48: 'l', 49: 'm', 50: 'n',
    51: 'o', 52: 'p', 53: 'q', 54: 'r', 55: 's', 56: 't', 57: 'u', 58: 'v', 59: 'w', 60: 'x',
    61: 'y', 62: 'z', 63: '-'}

"""
letters = { 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j',
11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't',
21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z', 27: '-'}
"""
wordChecker = SpellChecker()

# Load the models built in the previous steps
cnn_model = load_model('Senior_Project_cnn_model.h5')

def letterFinder(imageName):

	prediction = 62

	# Load the image and convert it to grayscale
	image = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)

	# Resize the image to model input dimension
	image = cv2.resize(image, (28, 28), 1)

	# Convert img to a numpy array
	image = np.array(image)

	# Preprocess the image (similar preprocessing was performed before training)
	image = image.astype('float32') / 255
	
	# Define prediction variables
	prediction = cnn_model.predict(image.reshape(1, 28, 28, 1))[0]

	print(prediction)

	prediction = np.argmax(prediction)

	print(prediction)


	return str(characters[int(prediction)])
