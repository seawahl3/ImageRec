import cv2
import os
from Main import FrameHandler
import shutil
import math
from natsort import natsorted, ns


def HandleVideo(Video):
    if not os.path.exists('VideoFrameData/'):
        os.mkdir('VideoFrameData/')

    if not os.path.exists('VideoFrameData/%s'%Video):
        os.mkdir('VideoFrameData/%s'%Video)
    
    cap = cv2.VideoCapture(Video)
    framerate = cap.get(cv2.CAP_PROP_FPS)
    currentFrame = 0
    frameCount = 0
    timeStamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
    
    ret, frame = cap.read() 
    while(cap.isOpened()):
        prev_frame=frame[:]
        ret, frame = cap.read() 
        if ret:
            currentFrame += 1
            # Frame after interval, 15 here means frame every 15 seconds
            if currentFrame >= (framerate * 10):
                currentFrame = 0
                frameCount += 1
                cv2.imwrite('VideoFrameData/%s/Frame%d.png'%(Video, frameCount), frame)
                timeStamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        else:
            # Get Last Frame
            frameCount += 1
            cv2.imwrite('VideoFrameData/%s/Frame%d.png'%(Video, frameCount),  prev_frame)
            timeStamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            break
    
    cap.release()
    #Milliseconds to seconds
    timeStamps = [x * 0.001 for x in timeStamps]
    index = 0
    result = []
    for(i, f) in enumerate(natsorted(os.listdir('VideoFrameData'))):
        for (x, y) in enumerate(natsorted(os.listdir('VideoFrameData/%s'%f))):
            print("Video Frame being processed" + (x, y))
            t = (FrameHandler('VideoFrameData/%s/%s'%(f, y)), math.floor(timeStamps[index]))
            result.append(t)
            shutil.rmtree('letters')
            index += 1
        shutil.rmtree('out')
    shutil.rmtree('VideoFrameData')
    print(result)
    return result


#HandleVideo('TestVideo2.0.mp4')

#Method call to the Word Splitter using the file location of the video frames


