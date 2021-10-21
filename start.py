import cv2 as cv
from matplotlib import image
import numpy as np
#for test
import matplotlib.pylab as plt

class StartVideo():
    def __init__(self) -> None:
        self.video_cap = cv.VideoCapture("output.avi") 
    
    def main_work(self):
        try:
            while (True):
                ok, frame = self.video_cap.read()  # read frame from video stream
                if ok:  # frame captured without any errors
                    cv.imshow('Frame', frame)
                    cv.waitKey(1)
        except KeyboardInterrupt:
            cv.destroyAllWindows()

class Test():
    def __init__(self) -> None:
        self.frame = cv.imread('color_sun_289.jpg')
        self.frame = cv.cvtColor(self.frame, cv.COLOR_BGR2RGB)
        height = self.frame.shape[0]
        width= self.frame.shape[1]
        self.region_interest_view = [
            (0, height//2),
            (width//2, height//2),
            (width, height),
            (0, height//2)
        ]
    def regionOfInterest(self, frame, vertices):
        mask = np.zeros_like(frame)
        #channel_count = frame.shape[2]
        match_mask_color = 200
        cv.fillPoly(mask, vertices, match_mask_color)
        masked_image = cv.bitwise_and(frame, mask)
        return masked_image

    def drow_lines(self, frame, lines):
        frame = np.copy(frame)
        line_image = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)
        
        frame = cv.addWeighted(frame, 0.8, line_image, 1, 0.0)
        return frame

        


if __name__ == "__main__":
    #video = StartVideo()
    #video.main_work()
    test = Test()
    print(test.frame.shape)
    gray_image = cv.cvtColor(test.frame, cv.COLOR_RGB2GRAY) 
    canny_image = cv.Canny(gray_image, 100, 200)
    croppe_image = test.regionOfInterest(canny_image,
                    np.array([test.region_interest_view], np.int32),)
    
    lines = cv.HoughLinesP(croppe_image, rho=6, theta=np.pi/60,
                            threshold=160, lines=np.array([]), 
                            minLineLength=20, maxLineGap=25)

    #image_with_lnes = test.drow_lines(test.frame, lines)
    plt.imshow(croppe_image)
    
    plt.show()
    #video = StartVideo()
    #video.main_work()

   

    
