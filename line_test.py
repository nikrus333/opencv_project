import cv2
import numpy as np
import subprocess as sp
import time
import atexit
import sys
import signal
import psutil



Speed = 50






frames = [] # stores the video sequence for the demo

# Video capture parameters
[w, h] = [640,480]  # Resolution
h = 480
#bytesPerFrame = w * h
fps = 40 # setting to 250 will request the maximum framerate possible

lateral_search = 20 # number of pixels to search the line border
start_height = h - 5 # Scan index row 235

# "raspividyuv" is the command that provides camera frames in YUV format
#  "--output -" specifies stdout as the output
#  "--timeout 0" specifies continuous video
#  "--luma" discards chroma channels, only luminance is sent through the pipeline
# see "raspividyuv --help" for more information on the parameters
#videoCmd = "raspividyuv -w "+str(w)+" -h "+str(h)+" --output - --timeout 0 -vs -co 50 -br 50 --framerate "+str(fps)+" --luma --nopreview"
#videoCmd = videoCmd.split() # Popen requires that each parameter is a separate string

#cameraProcess = sp.Popen(videoCmd, stdout = sp.PIPE) # start the camera
#atexit.register(cameraProcess.terminate) # this closes the camera process in case the python scripts exits unexpectedly

# wait for the first frame and discard it (only done to measure time more accurately)
#rawStream = cameraProcess.stdout.read(bytesPerFrame)

print("Recording...")
cap = cv2.VideoCapture(0)
no_points_count = 0

while(cap.isOpened()):
#for qwerty in xrange(500):
    ret, frame = cap.read()
    
   
    #if ret==True:
        #frame = cv2.flip(frame,0)

        # write the flipped frame


    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_rgb  = frame
    cv2.imshow('frame_rgb',frame_rgb)
    #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) # Drawing color points requires RGB image
    ret, thresh = cv2.threshold(frame, 105, 255, cv2.THRESH_BINARY)
    #tresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

    signed_thresh = thresh[start_height].astype(np.int16) # select only one row
    diff = np.diff(signed_thresh)   #The derivative of the start_height line

    points = np.where(np.logical_or(diff > 200, diff < -200)) #maximums and minimums of derivative

    cv2.line(frame_rgb,(0,start_height),(640,start_height),(0,255,0),1) # draw horizontal line where scanning 

    if len(points) > 0 and len(points[0]) > 1: # if finds something like a black line
       

        middle = (points[0][0] + points[0][1]) / 2
        print(middle)
        cv2.circle(frame_rgb, (points[0][0], start_height), 2, (255,0,0), -1)
        cv2.circle(frame_rgb, (points[0][1], start_height), 2, (255,0,0), -1)
        #cv2.circle(frame_rgb, (middle, start_height), 2, (0,0,255), -1)

        #print(int((middle-320)/int(sys.argv[1])))
    
    else:

        start_height -= 5
        start_height = start_height % h
        no_points_count += 1
        Speed -= 0.1
	           

    

    frames.append(frame_rgb)
    frames.append(thresh)	
    if psutil.virtual_memory().percent >= 85:
        del frames[0]

    if no_points_count > 50:
        print("Line lost")
        break

cleanup_finish()

def cleanup_finish():
   
    print("Writing frames to disk...")
    out = cv2.VideoWriter("drive_test.avi", cv2.cv.CV_FOURCC(*"MJPG"), 5, (w,h))

    for frame in frames:
        out.write(frame)

    out.release()

