import numpy as np

from matplotlib import pyplot as plt

import os
import glob as glob
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import 
import cv2
#import glob as globr
img_dirl = "/home/vidyaa/3D-2D_Stereo_VO/Data Set/2010_03_09_drive_0019/I1_*"
img_dirr = "/home/vidyaa/3D-2D_Stereo_VO/Data Set/2010_03_09_drive_0019/I2_*"# Enter Directory of all images 

filel = glob.glob(img_dirl)
filer = glob.glob(img_dirr)
filel.sort()
filer.sort()
i=0
for fl,fr in zip(filel,filer):
	print(fl)
	imgL = cv2.imread(fl)
	imgR = cv2.imread(fr)
	lnew = cv2.cvtColor(imgL, cv2.COLOR_RGB2GRAY)
	rnew = cv2.cvtColor(imgR, cv2.COLOR_RGB2GRAY)
	stereo = cv2.StereoBM_create(numDisparities=128, blockSize =15)
	disparity = stereo.compute(lnew,rnew)
	i=i+1
	#print(disparity)
	cv2.imwrite('/home/vidyaa/disparity1/%04d.png' %i, disparity)
	cv2.waitKey(0)

	
    
#stereo = cv2.StereoBM(cv2.STEREO_BM_BASIC_PRESET,ndisparities=16, SADWindowSize=15)
#	stereo = cv2.StereoBM_create(numDisparities=128, BlockSize =15)
