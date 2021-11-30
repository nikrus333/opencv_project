import cv2 as cv
import numpy as np
import math
class Detection():
    def __init__(self):

        self.frame = None

    def binareid(self, frame, bin_cof = 115): # бинаризация 
        r_channel = frame[:, :, 2]  
        binary = np.zeros_like(r_channel)
        binary[(r_channel > bin_cof)] = 1

        hls = cv.cvtColor(frame, cv.COLOR_BGR2HLS)
        s_channel = frame[:, :, 2]
        binary2 = np.zeros_like(s_channel)
        binary2[(s_channel > 110)] = 1

        all_binary = np.zeros_like(binary)
        all_binary[((binary == 1) | (binary2 == 1))] = 255
        return all_binary

if __name__ == "__main__":
    detection = Detection()
    hsv_min = np.array((2, 28, 65), np.uint8)
    hsv_max = np.array((26, 238, 255), np.uint8)
    frame = cv.imread('1.jpg')
    frame_bin = detection.binareid(frame)
    #hsv = cv.cvtColor( frame_bin, cv.COLOR_BGR2HSV ) # меняем цветовую модель с BGR на HSV 
    #thresh = cv.inRange( frame_bin, hsv_min, hsv_max ) # применяем цветовой фильтр
    # ищем контуры и складируем их в переменную contours
    _, contours, hierarchy = cv.findContours( frame_bin.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # отображаем контуры поверх изображения
    
    cv.imshow('bin', frame_bin)
    contours_new = []
    for i in range(len(contours)):
        area = cv.contourArea(contours[i], oriented = False)
        perometr = cv.arcLength(contours[i], True)
        if perometr > 300 and area > 600:
            solution = perometr / (math.sqrt(area))
            if solution > 3.52 and solution < 4:
                contours_new.append(contours[i])
                cv.drawContours( frame, contours, i, (255 - 5*i, 5 * i , 0), 3, cv.LINE_AA, hierarchy, 1 )
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

   
    #cv.drawContours( frame, contours_new, -1, (255,0,0), 3, cv.LINE_AA, hierarchy, 1 )
    while True:
        print(len(contours_new))
        cv.imshow('frame', frame) 
        if cv.waitKey(1) & 0xFF == ord('q'):
                break
