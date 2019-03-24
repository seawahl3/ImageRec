import shutil
import sys
import os
import pytesseract
from PIL import Image
from spellchecker import SpellChecker

# Global Variables
wordChecker = SpellChecker()


def findWord(folderpath):
    # Empty string to add each of the letters to as they are found
    word = ""
    for image in enumerate(os.listdir(folderpath)):
        word += letterIdentifier(folderpath + '\\' + image[1])

    return word


def letterIdentifier(imageName):
    img = Image.open(imageName)
    text = pytesseract.image_to_string(img, config='--psm 10 --dpi 70')
    return text
