import os 
import cv2
import numpy as np
import math

def wordSplitting(img, kernelSize=30, sigma=11, theta=7, minSize=0):
    kernel = createKernel(kernelSize, sigma, theta)
    imgFiltered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
    
    #cv2.imshow('image',imgFiltered)
    #cv2.waitKey(0)
    
    (_, imgThres) = cv2.threshold(imgFiltered, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    imgThresh = 255 - imgThres
    
    #cv2.imshow('image',imgThresh)
    #cv2.waitKey(0)
        
        
    (_, components, _) = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    res = []
    for c in components:
        if cv2.contourArea(c) < minSize:
            continue
        currBox = cv2.boundingRect(c)
        (x, y, w, h) = currBox
        currImg = img[y:y+h, x:x+w]
        res.append((currBox, currImg))
        
    return sorted(res, key=lambda entry:entry[0][0])
    
def convertImage(img, height):
    assert img.ndim in (2, 3)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = img.shape[0]
    factor = height 
    return img #cv2.resize(img, dsize=None, fx=factor, fy=factor)
    


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


def main():
    
    print(os.getcwd())
    imgFile = os.listdir('PictureData')
    
    for(i, f) in enumerate(imgFile):
        print('File: %s is being processed for words'%f)
        
        img = convertImage(cv2.imread('PictureData/%s'%f), 100)
        
        #Kernel Size, sigma and theta need to be odd
        # Words : kernel:21 sigma:15 theta:9 minSize:250
        # Letters: kernel:1 sigma:1 theta:1 minSize:0
        res = wordSplitting(img, kernelSize=21, sigma=15, theta=9, minSize=250)
        
        if not os.path.exists('out/%s'%f):
            os.mkdir('out/%s'%f)
            
        print('Found %d'%len(res) + ' word(s) in %s'%f)
        for (j, w) in enumerate(res):
            (wordBox, wordImg) = w
            (x, y, w, h) = wordBox
            cv2.imwrite('out/%s/%d.png'%(f, j), wordImg)
            cv2.rectangle(img, (x,y), (x+w, y+h), 0, 1)
            cv2.imwrite('out/%s/summary.png'%f, img)
            
            
    letterFile = os.listdir('out')      
    for(i, f) in enumerate(letterFile):
        print('File: %s is being processed for letters' %f)
        
        img = convertImage(cv2.imread('out/%s'%f), 100)
        
        if not os.path.exists('letters/%s' %f):
            os.mkdir('letters/%s'%f)
            
        res = wordSplitting(letterFile, kernelSize =1, sigma=1, theta=1, minSize=0)
        
        print('Found %d'%len(res) + 'letter(s) in %s'%f)
        for(j, w) in enumerate(res):
            (wordBox, wordImg) = w
            (x, y, w, h) = wordBox
            cv2.imwrite('letters/%s/%d.png'%(f, j), wordImg)
            cv2.rectangle(img, (x,y), (x+w, y+h), 0, 1)
            cv2.imwrite('letters/%s/summary.png'%f, img)
        # call erics code
            
        
            
if __name__ == '__main__':
    main()
    