#!/usr/bin/python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from time import sleep
from std_msgs.msg import Float64, String
pub_image = rospy.Publisher('/image/line', Image, queue_size=1)
pub_image_test = rospy.Publisher('/image/line/test', Image, queue_size=1)

pub_error = rospy.Publisher('error_lane', Float64, queue_size=1)
# pub_white_error = rospy.Publisher('line_white_error', Float64, queue_size=1)
cvBridge = CvBridge()
top_x = 240
top_y = 240
bottom_x = 330
bottom_y = 240
stop_sign = False
counter = 0
def cbImageProjection(data):
	global counter
	if counter != 3 :
		counter += 1
		return
	else:
		counter = 0
	cv_image_original = cvBridge.imgmsg_to_cv2(data, "bgr8")
	pts_src = np.array([[320 - top_x, 240 - top_y], [320 + top_x, 240 - top_y],[320 + bottom_x, 240 + bottom_y], [320 - bottom_x, 240 + bottom_y]])
	# pub_image.publish(cvBridge.cv2_to_imgmsg(cv_image_original, "bgr8"))

	# selecting 4 points from image that will be transformed
	pts_dst = np.array([[0, 0], [640, 0], [640, 480], [0, 480]])
	# for point in pts_src:
    # cv2.circle(cv_image_original,(point[0], point[1]), 5, (0,0,255), -1)
	# finding homography matrix
	h, status = cv2.findHomography(pts_src, pts_dst)
	# homography process
	cv_image_homography = cv2.warpPerspective(cv_image_original, h, (640, 480))
	# cv_image_homography = cv_image_original
	# print(cv_image_original.shape)
	cv_image_homography = cv2.medianBlur(cv_image_homography, 5)
	# cv_image_homography = cv_image_original#homography(cv_image_original)
	yellow_array = mask_yellow(cv_image_homography)
	white_array = mask_white(cv_image_homography)
	detected = cv_image_homography

	if(len(yellow_array) > 0):
		point_before = yellow_array[0]
		for point in yellow_array:
			detected = cv2.line(detected, (point[0], point[1]), (point_before[0],point_before[1]), (0,255,255),8)
			point_before = point

	if(len(white_array) > 0):
		point_before = white_array[0]
		for point in white_array:
			detected = cv2.line(detected, (point[0], point[1]), (point_before[0],point_before[1]), (255,255,255),8)
			point_before = point

	pub_image.publish(cvBridge.cv2_to_imgmsg(detected, "bgr8"))
	error = Float64()
	error.data = calculate_error(yellow_array, white_array)
	pub_error.publish(error)
	# pub_white_error.publish(error)
	counter = 0

	
def mask_yellow(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	# define range of yellow color in HSV
	lower_yellow = np.array([15, 70, 100])  # 0,100,100
	upper_yellow = np.array([70, 255, 255])  # 30,255,255
	# Threshold the HSV image to get only yellow colors
	mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	# Bitwise-AND mask and original image
	res = img
	# res = cv2.bitwise_and(img, img, mask = mask)
	fraction_num = np.count_nonzero(mask)
	# pub_image_test.publish(cvBridge.cv2_to_imgmsg(mask, "8UC1"))
	point_arr = []
	stop_flag = False
	if fraction_num > 1000:
		k = 0
		jold = 0
		for i in range(mask.shape[0]-1,0,-10):
			if stop_flag == True:
				break
			for j in range(0,int(mask.shape[1]/2),10):
				if mask[i,j] > 0:
					point_arr.append([j,i])
					k+=1
					if abs(j-jold) > 40 and k > 1:
						point_arr.pop()
						stop_flag = True
					jold = j
					break
		# if(len(point_arr) > 0):
			# point_before = point_arr[0]
			# for point in point_arr:
				# res = cv2.line(res, (point[0], point[1]), (point_before[0],point_before[1]), (0,0,255),8)
				# point_before = point
	return point_arr
	

def mask_white(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	sensitivity = 140  # range of sensitivity=[90,150]
	lower_white = np.array([60, 10, 255-sensitivity])
	upper_white = np.array([255, sensitivity, 255])
# Threshold the HSV image to get only yellow colors
	
	mask = cv2.inRange(hsv, lower_white, upper_white)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
	pub_image_test.publish(cvBridge.cv2_to_imgmsg(mask, "8UC1"))

	# Bitwise-AND mask and original image
	# res = cv2.bitwise_and(img, img, mask = mask)
	res = img
	fraction_num = np.count_nonzero(mask)
	point_arr = []
	stop_flag = False

	if fraction_num > 1000:
		k = 0
		jold = 0
		for i in range(mask.shape[0]-1,0,-10): #Y
			if stop_flag == True:
				break
			for j in range(mask.shape[1]-1,int(320),-10): #x
				if mask[i,j] > 0:
					point_arr.append([j,i])
					k+=1
					if abs(j-jold) > 40 and k > 1:
						point_arr.pop()
						stop_flag = True
					jold = j
					break
		# if len(point_arr) > 0:
			# point_before = point_arr[0]
			# for point in point_arr:
				# res = cv2.line(res, (point[0], point[1]), (point_before[0],point_before[1]), (0,0,255),8)
				# point_before = point
	return point_arr
def calculate_error(yellow_array, white_array):
	global stop_sign
	error_yell = 0
	error_white = 0
	weight = 0
	i = 1
	for yel in yellow_array:
		#when yel[2] = 600 then weight = 0 and if yel[2] = 0 wheight = 1
		weight = yel[1]*0.0017 + 1
		if(stop_sign == False):
			error_yell = weight*(60 - yel[0]) + error_yell
		else:
			error_yell = 0
		i+=1
	error_yell = error_yell/i
	for white in white_array:
		weight = white[1]*0.0017 + 1
		if(stop_sign == False):
			error_white = weight*(600 - white[0]) + error_white
		else:
			error_white = weight*(270 - white[0]) + error_white
		i+=1
	error_white = error_white/i
	# print("white "+ str(error_white) + " yellow "+ str(error_yell))
	if abs(error_white) < 20:
		return error_yell
	elif abs(error_yell) < 20:
		return error_white
	else:
		return (error_white + error_yell)/2

def cb_sign(data):
	global stop_sign
	if(data.data == "stop"):
		stop_sign = True
if __name__ == '__main__':
	rospy.init_node('line_detect')
	sub_image = rospy.Subscriber('image', Image, cbImageProjection, queue_size=1)
	sub_sign = rospy.Subscriber('sign', String, cb_sign, queue_size=1)
	while not rospy.is_shutdown():
		try:
			rospy.sleep(0.1)
		except KeyboardInterrupt:
			break
			cv2.destroyAllWindows()
