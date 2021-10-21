import cv2


# cap = cv2.VideoCapture(
#     "rtspsrc location=rtsp://192.168.42.1/cam latency=50 buffer-mode=slave ! decodebin ! videoconvert ! appsink",
#     cv2.CAP_GSTREAMER,
# )
import numpy as np


cap = cv2.VideoCapture("rtspsrc location=rtsp://192.168.42.1/cam latency=50 buffer-mode=slave ! decodebin ! videoconvert ! appsink",
    cv2.CAP_GSTREAMER,)

# Define the codec and create VideoWriter object
#fourcc = cv2.cv.CV_FOURCC(*'DIVX')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
out = cv2.VideoWriter("output.avi",
cv2.VideoWriter_fourcc(*"MJPG"), 30,(640,480))
count = 1
while(cap.isOpened()):
    ret, frame = cap.read()
    
    print(count)
    count = count + 1
    if count > 4 :
        break
    if ret==True:
        frame = cv2.flip(frame,0)

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
       
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()