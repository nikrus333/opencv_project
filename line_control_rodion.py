#!/usr/bin/python3
# -*- coding: utf-8 -*-

import rospy
from time import sleep
from std_msgs.msg import Float64, Bool
from geometry_msgs.msg import Twist
pub_vel = rospy.Publisher('cmd_vel', Twist, queue_size=1)
integral = 0
move_flag = True
def cbError(error):
	global integral, move_flag
	if(move_flag == True):
		velocity = Twist()
		integral = integral + 0.000005*error.data
		proportional = 0.005*error.data
		up = proportional+integral
		velocity.angular.z = up
		velocity.linear.x = 0.12 - 0.09*abs(up)
		# print(up)
		pub_vel.publish(velocity)
def cb_flag(data):
	global move_flag
	move_flag = data.data

if __name__ == '__main__':
	rospy.init_node('line_control')
	sub_image = rospy.Subscriber('error_lane', Float64, cbError, queue_size=1)
	sub_move_flag = rospy.Subscriber('line_move_flag', Bool, cb_flag, queue_size=1)
	while not rospy.is_shutdown():
		try:
			rospy.sleep(0.1)
		except KeyboardInterrupt:
			break
			