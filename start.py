import cv2 as cv

import numpy as np
#for test
import matplotlib.pylab as plt
from numpy.core.defchararray import center
from numpy.lib.histograms import histogram

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

class Solution():
    def __init__(self) -> None:
        pass
    def satartData(self, frame):
        blank_i = 1

    def binareid(self, frame, bin_cof = 235): # бинаризация 
        r_channel = frame[:, :, 2]  
        binary = np.zeros_like(r_channel)
        binary[(r_channel > bin_cof)] = 1

        hls = cv.cvtColor(frame, cv.COLOR_BGR2HLS)
        s_channel = frame[:, :, 2]
        binary2 = np.zeros_like(s_channel)
        binary2[(s_channel > 230)] = 1

        all_binary = np.zeros_like(binary)
        all_binary[((binary == 1) | (binary2 == 1))] = 255
        return all_binary

    def transformFunc(self, frame):   # аффинные преобразования
        src = np.float32([  
                            [639, 470],
                            [620, 59] ,
                            [20, 58],
                            [1, 470]
                            ])
        dot_src = np.int32(src, dtype = np.int32)
        image_size = [480, 640]                   
        dst = np.float32([  [image_size[1], image_size[0]],
                            [image_size[1], 0],
                            [0, 0],
                            [0, image_size[0]]
                            
                            
                                                    ])
        cv.polylines(frame, [dot_src], True, 200)
        m_transform = cv.getPerspectiveTransform(src, dst)
        frame = cv.warpPerspective(frame, m_transform, (image_size[1], image_size[0]), flags=cv.INTER_LINEAR)
        return frame

    def foundWhiteTable(self, frame):  # поиск беллых линий и отрисовка их 
        histogram = np.sum(frame[frame.shape[0] // 2 :, :], axis=0)
        midpoint = histogram.shape[0] // 2
        index_left = np.argmax(histogram[:midpoint])
        index_right = np.argmax(histogram[midpoint:]) + midpoint
        warped_visual = frame.copy()
        cv.line(warped_visual, (index_left, 0), (index_left, warped_visual.shape[0]), 110, 2) # отрисовка линий, тестовое потом удалить 
        cv.line(warped_visual, (index_right, 0), (index_right, warped_visual.shape[0]), 110, 2)
        
        nwindows = 9
        window_height = np.int(frame.shape[0] / nwindows)
        window_half_height = 30
        x_center_left_windows = index_left
        x_center_right_windows = index_right
        left_lane_ind = np.array([], dtype=np.int16)
        right_lane_ind = np.array([], dtype=np.int16)

        out_frame = np.dstack((frame, frame, frame)) 
        non_zero = frame.nonzero()
        white_pixel_ind_y = np.array(non_zero[0])
        white_pixel_ind_x = np.array(non_zero[1])
        for window in range(nwindows):
            win_y1 = frame.shape[0] - (window + 1) * window_height
            win_y2 = frame.shape[0] - (window) * window_height
            left_win_x1 = x_center_left_windows - window_half_height
            left_win_x2 = x_center_left_windows + window_half_height
            right_win_x1 = x_center_right_windows - window_half_height
            right_win_x2 = x_center_right_windows + window_half_height
        
            cv.rectangle(out_frame, (left_win_x1, win_y1), (left_win_x2, win_y2), (50 + window * 21, 0, 0), 2)
            cv.rectangle(out_frame, (right_win_x1, win_y1), (right_win_x2, win_y2), (50 + window * 21, 50 + window * 21, 0), 2)

            good_left_ind = ((white_pixel_ind_y >= win_y1) & (white_pixel_ind_y <= win_y2) & 
                (white_pixel_ind_x >=left_win_x1) & (white_pixel_ind_x <= left_win_x2)).nonzero()[0]
            good_right_ind = ((white_pixel_ind_y >= win_y1) & (white_pixel_ind_y <= win_y2) & 
                (white_pixel_ind_x >=right_win_x1) & (white_pixel_ind_x <= right_win_x2)).nonzero()[0]
            
            left_lane_ind = np.concatenate((left_lane_ind, good_left_ind))
            right_lane_ind = np.concatenate((right_lane_ind, good_right_ind))

            if len(good_left_ind) > 50:
                x_center_left_windows = np.int(np.mean(white_pixel_ind_x[good_left_ind]))
            if len(good_right_ind) > 50:
                x_center_right_windows = np.int(np.mean(white_pixel_ind_x[good_right_ind]))

           

        out_frame[white_pixel_ind_y[left_lane_ind], white_pixel_ind_x[left_lane_ind]] = [255, 0, 0]
        out_frame[white_pixel_ind_y[right_lane_ind], white_pixel_ind_x[right_lane_ind]] = [255, 255, 0]

        #def solution_center_line(out_frame, leftx, lefty )
        leftx = white_pixel_ind_x[left_lane_ind]
        lefty = white_pixel_ind_y[left_lane_ind] 
        rightx = white_pixel_ind_x[right_lane_ind]
        righty = white_pixel_ind_y[right_lane_ind]
        try:
            left_fit = np.polyfit(lefty, leftx, 2) 
            right_fit = np.polyfit(righty, rightx, 2)      
            center_fit = ((left_fit + right_fit) / 2)
        
            for ver_ind in range(out_frame.shape[0]):
                gor_ind = ((center_fit[0]) * (ver_ind ** 2) + 
                            center_fit[1] * ver_ind +
                            center_fit[2])
                cv.circle(out_frame, (int(gor_ind), int(ver_ind)), 2, (255, 0, 255),1)
        except:
            pass
        return out_frame

    def hough_transform(self, frame):
    
        rho = 1              #Distance resolution of the accumulator in pixels.
        theta = np.pi/180    #Angle resolution of the accumulator in radians.
        threshold = 20       #Only lines that are greater than threshold will be returned.
        minLineLength = 20   #Line segments shorter than that are rejected.
        maxLineGap = 300     #Maximum allowed gap between points on the same line to link them
        frame =  cv.HoughLinesP(frame, rho = rho, theta = theta, threshold = threshold,
                            minLineLength = minLineLength, maxLineGap = maxLineGap)
        return frame

if __name__ == "__main__":

    cap = cv.VideoCapture(0)
    sol = Solution()

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            original_frame = frame
            cv.imshow('frame_original',frame)
            frame_bin = sol.binareid(frame)
            frame_transofrm = sol.transformFunc(frame_bin)
            cv.imshow('frame_transofrm',frame_transofrm)
            frame_hough = sol.hough_transform(frame_transofrm)
            frame_white_table =  sol.foundWhiteTable(frame_bin)
            cv.imshow('frame_bin',frame_bin)
            cv.imshow('frame_white_table',frame_white_table)
            #cv.imshow('frame_trans',frame_transofrm)
            #cv.imshow('frame_trans',frame_hough)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        
        else:
            break
    
    cap.release()

    cv.destroyAllWindows()

# Release everything if job is finished



   

    
