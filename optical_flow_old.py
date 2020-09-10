import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import glob as glob
img_dir = "/home/vidyaa/disparity/*"


file = glob.glob(img_dir)
file.sort()
i=0
nPts=40
prev = cv2.imread("/home/vidyaa/disparity/0001.png")
for f in file:
	print(f)
	img = cv2.imread(f)

	fast = cv2.FastFeatureDetector_create()

	H,W,n = img.shape
	kp = []
	Tile_H = 20
	Tile_W = 20
	nFeatures=1
	for y in range(0, H, Tile_H):
		for x in range(0, W, Tile_W):
			Patch_Img = img[y:y+Tile_H, x:x+Tile_W]
			keypoints = fast.detect(Patch_Img,None)
			for pt in keypoints:
				pt.pt = (pt.pt[0] + x, pt.pt[1] + y)

			if (len(keypoints) > nFeatures):
				keypoints = sorted(keypoints,key=lambda x: -x.response)
				for kpt in keypoints[0:nFeatures]:
					kp.append(kpt)
			else:
				for kpt in keypoints:
					kp.append(kpt)
	Img = cv2.drawKeypoints(img, kp, None, color=(255,0,0))
	cv2.imwrite('/home/vidyaa/binning/%d.png'%i,Img)
	trackPts = cv2.KeyPoint_convert(kp)
	trackPts = np.expand_dims(trackPts, axis=1)
	lk_params = dict( winSize  = (15,15),maxLevel = 3,\
                      criteria = (cv2.TERM_CRITERIA_EPS | \
                      cv2.TERM_CRITERIA_COUNT, 50, 0.03))
	Pts_2, st, err = cv2.calcOpticalFlowPyrLK(\
                     prev,img, trackPts, None,\
                     flags=cv2.MOTION_AFFINE, **lk_params)

	# separate points that were tracked successfully
	ptTrackable = np.where(st == 1, 1,0).astype(bool)
	TrkPts_1 = trackPts[ptTrackable, ...]
	TrkPts_2 = Pts_2[ptTrackable, ...]
	TrkPts_2 = np.around(TrkPts_2)

	print ("Points successfully tracked: " + str(len(Pts_2)))

	error = 4
	errTrackablePts = err[ptTrackable, ...]
	errThreshPts = np.where(errTrackablePts < \
                            error, 1, 0).astype(bool)
	# Dynamically change threshold to get required points
	while np.count_nonzero(errThreshPts) > nPts:
		error = round(error - 0.1,1)
		errThreshPts = np.where(errTrackablePts < \
                                error, 1, 0).astype(bool)

	while np.count_nonzero(errThreshPts) < nPts :
		error = round(error + 0.1,1)
		errThreshPts = np.where(errTrackablePts < \
                                error, 1, 0).astype(bool)
	if error >= 8:
		print ("Max Limit Reached... Exiting loop")
		

	TrkPts_1 = TrkPts_1[errThreshPts, ...]
	TrkPts_2 = TrkPts_2[errThreshPts, ...]
	print ("Points with error less than " \
               + str(error) + " : " + str(len(TrkPts_1)))


	prev = img
print(trackPts)

