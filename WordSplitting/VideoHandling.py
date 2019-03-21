import cv2
import os



def HandleVideo(Video):
    
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
            #cv2.imshow('frame',frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
                #break
            #Get a frame every 30 seconds
            currentFrame += 1
            if currentFrame == (framerate * 30):
                currentFrame = 0
                frameCount += 1
                #cv2.imshow('frame', frame)
                cv2.imwrite('VideoFrameData/%s/Frame%d.png'%(Video, frameCount), frame)
                timeStamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        else:
            frameCount += 1
            cv2.imwrite('VideoFrameData/%s/Frame%d.png'%(Video, frameCount),  prev_frame)
            timeStamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            break
    
    cap.release()
    #Milliseconds to seconds
    timeStamps = [x * 0.001 for x in timeStamps]
    print(timeStamps)
        


HandleVideo('TestVideo2.0.mp4')

#Method call to the Word Splitter using the file location of the video frames


