import numpy as np
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import 
import cv2
from matplotlib import pyplot as plt
import os
import glob as glob
img_dir = "/home/vidyaa/left/*"


file = glob.glob(img_dir)
file.sort()
i=0
for f in file:
	print(f)
	img = cv2.imread(f)

	fast = cv2.ORB_create()
	# find and draw the keypoints
	kp = fast.detect(img,None)
	#print(kp)
	img2 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))
	'''
	# Print all default params
	print( "Threshold: {}".format(fast.getThreshold()) )
	print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
	print( "neighborhood: {}".format(fast.getType()) )
	print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )

	i=i+1
	cv2.imwrite('/home/vidyaa/fastnms/%d.png'%i,img2)

	# Disable nonmaxSuppression
	fast.setNonmaxSuppression(0)
	kp = fast.detect(img,None)

	print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )

	img3 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))
	'''
	cv2.imwrite('/home/vidyaa/sift/%d.png'%i,img2)
	cv2.waitKey(0)
'''
# Print all default params
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
cv.imwrite('fast_true.png',img2)
# Disable nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(img,None)
print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
img3 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
cv.imwrite('fast_false.png',img3)
'''
