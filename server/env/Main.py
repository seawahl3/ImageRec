import os 
import cv2
import numpy as np
import math
from ImageToText import findWord
from natsort import natsorted, ns


# Title: WordSegmentation
# Author: Harald Scheidl
# Date: 2018
# Availability: https://github.com/githubharald/WordSegmentation
def Splitter(img, kernelSize=30, sigma=11, theta=7, minSize=0, maxSize=0):
    #Creates blurring kernel
    kernel = createKernel(kernelSize, sigma, theta)
    #Blurrs passed image
    imgFiltered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
    
    # Uncomment to show blurred image
    #cv2.imshow('image',imgFiltered)
    #cv2.waitKey(0)
    
    cv2.imwrite('FilteredPictures/Blurred.png', imgFiltered)
    
    imgFiltered = cv2.equalizeHist(imgFiltered)
    #cv2.imshow('image',imgFiltered)
    #cv2.waitKey(0)
    
    
    
    #Black-white Threshold
    (_, imgThresh) = cv2.threshold(imgFiltered, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    imgThresh = 255 - imgThresh
    
    cv2.imwrite('FilteredPictures/Blackwhite.png', imgThresh)

    # Uncomment to show image with black-white threshold
    #cv2.imshow('image',imgThresh)
    #cv2.waitKey(0)
    

        
    #Finds the contours of each element to be split
    (_, components, _) = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #Checks for min and max size of contours, then adds padding to contours to give wiggle room around the split elements
    res = []
    for c in components:
        if cv2.contourArea(c) < minSize:
            continue
        if cv2.contourArea(c) > maxSize:
            continue
        currBox = cv2.boundingRect(c)
        xPadding = 3
        yPadding = 3
        (x, y, w, h) = currBox
        while x-xPadding < 0:
            xPadding -= 1
        while y-yPadding < 0:
            yPadding -= 1
        currImg = img[y-yPadding:y+h+yPadding, x-xPadding:x+w+xPadding]
        res.append((currBox, currImg))
        
    return sorted(res, key=lambda entry:entry[0][0])
    
def convertImage(img):
    assert img.ndim in (2, 3)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
    
def createKernel(kernelSize, sigma, theta):
    assert kernelSize % 2
    halfSize = kernelSize // 2 
    
    kernel = np.zeros([kernelSize, kernelSize])
    sigmaX = sigma
    sigmaY = sigma * theta
    
    for i in range(kernelSize):
        for j in range(kernelSize):
                x = i - halfSize
                y = j - halfSize
                
                expTerm = np.exp(-x**2 / (2 * sigmaX) - y**2 / (2*sigmaY))
                xTerm = (x**2 - sigmaX**2) / (2 * math.pi * sigmaX**5 * sigmaY)
                yTerm = (y**2 - sigmaY**2) / (2 * math.pi * sigmaY**5 * sigmaX)
                
                kernel[i, j] = (xTerm + yTerm) * expTerm
    kernel = kernel / np.sum(kernel)
    
    #print(kernel)
    return kernel


#Handles frames to be split into words then letters
def FrameHandler(frame):
    
    print(os.getcwd())
    
    print('File: ' + frame + ' is being processed for words')
    
    img = convertImage(cv2.imread(frame))
    
    #Kernel Size, sigma and theta need to be odd
    # Words : kernel:21 sigma:15 theta:9 minSize:250
    # Letters: kernel:1 sigma:1 theta:1 minSize:0
    print(type(img))
    res = Splitter(img, kernelSize=21,sigma=15, theta=11, minSize=100, maxSize=10000)
    
    if not os.path.exists('out'):
        os.mkdir('out')

    print('Found %d'%len(res) + ' word(s) ' + frame)
    for (j, w) in enumerate(res):
        (wordBox, wordImg) = w
        (x, y, w, h) = wordBox
        cv2.imwrite('out/%d.png'%j, wordImg)
        cv2.rectangle(img, (x,y), (x+w, y+h), 0, 1)
        cv2.imwrite('out/summary.png', img)
            
            
    letterFile = os.listdir('out')      
    for(i, f) in enumerate(letterFile):
        print('File: %s is being processed for letters ' %f)
        
        img = convertImage(cv2.imread('out/%s'%f))

        if not os.path.exists('letters'):
            os.mkdir('letters')
        
        if not os.path.exists('letters/%s' %f):
            os.mkdir('letters/%s'%f)
            
        res = Splitter(img, kernelSize =1, sigma=1, theta=1, minSize=1, maxSize=10000)
        
        print('Found %d'%len(res) + ' letter(s) in %s'%f)
        for(j, w) in enumerate(res):
            (wordBox, wordImg) = w
            (x, y, w, h) = wordBox
            cv2.imwrite('letters/%s/%d.png'%(f, j), wordImg)
            #cv2.rectangle(img, (x,y), (x+w, y+h), 0, 1)
            #cv2.imwrite('letters/%s/summary.png'%f, img)

    line = ''
    
    for path in enumerate(natsorted(os.listdir('letters'))):
        print(path)
        line += findWord('letters/' + path[1]) + ' '
    print(line)
    return line

            
        
            
#FrameHandler("image0.jpg")
    