import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt
import os
import glob as glob
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import 
import cv2
img_dir = "/home/vidyaa/disparity/*"
img_dirl = "/home/vidyaa/left/*"
from scipy.optimize import least_squares
import math as m
from math import *
file = glob.glob(img_dir)
filel = glob.glob(img_dirl)
file.sort()
filel.sort()
i=0
nPts=40
prev = cv2.imread("/home/vidyaa/disparity/0001.png")
prevl = cv2.imread("/home/vidyaa/left/0000000000.png")
scale=16
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
out = cv2.VideoWriter("Video_successful.avi",\
                            fourcc, 30.0, (748,282))
Calib_file = open("/home/vidyaa/Downloads/2011_09_26/calib_cam_to_cam.txt",'r')
Calib_file_lines = Calib_file.readlines()
P1_roi = np.zeros(12)
P2_roi = np.zeros(12)
for j in range(12):
    P1_roi[j] = Calib_file_lines[9].split()[j+1]
    P2_roi[j] = Calib_file_lines[17].split()[j+1]
P1_roi = P1_roi.reshape(3,4)
P2_roi = P2_roi.reshape(3,4)
# Setting the camera parameters
f = P1_roi[0,0]		
base = -P2_roi[0,3]/P2_roi[0,0]			
cx = P1_roi[0,2]				
cy = P1_roi[1,2]

GT_file = open('/home/vidyaa/insdata.txt','r')
GT_file_lines = GT_file.readlines()
StartGT = [0,0]
CurGT = [0,0]
StartGT[0] = float(GT_file_lines[0].split()[4])
StartGT[1] = float(GT_file_lines[0].split()[5])
Map = np.zeros((20,20,3), np.uint8)
centre = int(Map.shape[0]/2)
cv2.circle(Map,(centre,centre),5,color=(0,255,0), thickness = -1)
Position = [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0]]
# Getting the initial pose of the vehicle
# So that VO can be compared with Ground Truth
CurGT[0] = float(GT_file_lines[1].split()[4])
CurGT[1] = float(GT_file_lines[1].split()[5])
GT_x = CurGT[0] - StartGT[0]
GT_y = CurGT[1] - StartGT[1]
theta = m.atan2(-GT_x,-GT_y)
IniRot = [[m.cos(theta) , 0, m.sin(theta), 0],
          [0            , 1, 0           , 0],
          [-m.sin(theta), 0, m.cos(theta), 0],
          [0            , 0, 0           , 1]]
Position = np.dot(Position,IniRot)

select_type=input("select inlier or outlier(ID/OR):")

ID=0
def genEulerZXZMatrix(psi, theta, sigma):
    c1 = cos(psi)
    s1 = sin(psi)
    c2 = cos(theta)
    s2 = sin(theta)
    c3 = cos(sigma)
    s3 = sin(sigma)

    mat = np.zeros((3,3))

    mat[0,0] = (c1 * c3) - (s1 * c2 * s3)
    mat[0,1] = (-c1 * s3) - (s1 * c2 * c3)
    mat[0,2] = (s1 * s2)
    mat[1,0] = (s1 * c3) + (c1 * c2 * s3)
    mat[1,1] = (-s1 * s3) + (c1 * c2 * c3)
    mat[1,2] = (-c1 * s2)
    mat[2,0] = (s2 * s3)
    mat[2,1] = (s2 * c3)
    mat[2,2] = c2

    return mat

def mini(dof,random_3d_1, random_3d_2, random_2d_1, random_2d_2,P):
    Rmat = genEulerZXZMatrix(dof[0], dof[1], dof[2])
    translationArray = np.array([[dof[3]], [dof[4]], [dof[5]]])
    temp = np.hstack((Rmat, translationArray))
    perspectiveProj = np.vstack((temp, [0, 0, 0, 1]))
    forward = np.matmul(P,perspectiveProj)
    backward = np.matmul(P, np.linalg.inv(perspectiveProj))
    numPoints = len(random_2d_1)
    errorA = np.zeros((numPoints,3))
    errorB = np.zeros((numPoints,3))
    pred2d_1 =[]
    pred2d_2 =[]
    for i in range(len(random_3d_1)):
    	pred2d_1.append(np.matmul(forward,random_3d_2[i]))
    	pred2d_1[i] = pred2d_1[i]/pred2d_1[i][-1]
    	pred2d_2.append(np.matmul(backward,random_3d_1[i]))
    	pred2d_2[i] = pred2d_2[i]/pred2d_2[i][-1]
    	error_1 = random_2d_1[i]-pred2d_1[i]
    	error_2 = random_2d_2[i]-pred2d_2[i]
    	errorA[i,:] = error_1.reshape(1,3)[0]
    	errorB[i,:] = error_2.reshape(1,3)[0]
    residual = np.vstack((errorA,errorB))
    return residual.flatten()

#function for inlier detection
def find_bestPts_ID(point_cloud1,point_cloud2,minReq) :
    dist_difference = 0.05
    max_node = -1
    max_count = 0
    point_cloud1 = np.asarray(point_cloud1)
    point_cloud2 = np.asarray(point_cloud2)
    
    num_points = point_cloud1.shape[0]
    W = np.zeros((num_points,num_points))
    count = 0
    point_clouds_relative_dist = np.zeros((num_points,num_points))
    while max_node == -1:
        for i in range(num_points) : 
            diff_nodes_t1 = point_cloud1 - point_cloud1[i,:]
            diff_nodes_t2 = point_cloud2 - point_cloud2[i,:]
            dist_nodes_t1 = np.linalg.norm(diff_nodes_t1,axis=1)
            dist_nodes_t2 = np.linalg.norm(diff_nodes_t2,axis=1)
            abs_dist = abs(dist_nodes_t1 - dist_nodes_t2)

            point_clouds_relative_dist[i] = \
                np.asarray(abs_dist).T 
            wIdx = np.where(abs_dist < dist_difference)
            W[i,wIdx[0]] = 1
            count = np.sum(W[i,:])
            if count > max_count: 
                max_count = count
                max_node = i
        if max_count < minReq and dist_difference < 0.5 :
            max_count = 0
            max_node = -1
        if max_node == -1:
            dist_difference += 0.01
    count = 0
    clique = [max_node]

    while True :
        max_count = 0
        max_node = 0
        potentialnodes = list()
        Wsub = W[clique,:]
	# print(Wsub)
        for i in range(num_points) : 
            sumclique = np.sum(Wsub[:,i])
            if sumclique == len(clique) : 
                isin = True
            else : 
                isin = False
            if isin == True and i not in clique : 
                potentialnodes.append(i)
        max_count = 0
        max_node = 0 
        for i in range(len(potentialnodes)) : 
            Wsub = W[potentialnodes[i],potentialnodes]
            sumclique = np.sum(Wsub)
            if sumclique > max_count : 
                max_count = sumclique
                max_node = potentialnodes[i]

        if max_count == 0 :
            if len(clique) >= minReq : 
                break
            else :
                dist_difference += 0.05
                for k in range(num_points) : 
                    diff_nodes_t1 = point_cloud1 \
                                    - point_cloud1[k,:]
                    diff_nodes_t2 = point_cloud2 \
                                    - point_cloud2[k,:]
                    dist_nodes_t1 = \
                        np.linalg.norm(diff_nodes_t1,axis=1)
                    dist_nodes_t2 = \
                        np.linalg.norm(diff_nodes_t2,axis=1)
                    abs_dist = abs(dist_nodes_t1 - dist_nodes_t2)
                    point_clouds_relative_dist[k] = \
                        np.asarray(abs_dist).T 
                    wIdx = np.where(abs_dist < dist_difference)
                    W[k,wIdx[0]] = 1

        if len(clique) >= minReq or dist_difference > 10  :
            break
        clique.append(max_node)
    return clique

#main 
for focus,focusl in zip(file,filel):
	print(focus)
	img = cv2.imread(focus)
	imgl = cv2.imread(focusl)
	#feature_detection using FAST algorithm
	fast = cv2.FastFeatureDetector_create()
	#normalizing image for visualization
	disp_view = np.int16(img)
	disp_view = cv2.normalize(disp_view, None, beta=0,\
                alpha=np.amax(img)/16, norm_type=cv2.NORM_MINMAX);
	disp_view = np.uint8(disp_view)
	img_mask = cv2.inRange(disp_view,int(f*base/25),int(f*base/10))
	rows = img_mask.shape[0]
	cols = img_mask.shape[1]
	img_mask[int(3*rows/4):,:] = 0
	img_mask[:int(1*rows/10),:] = 0
	img_mask[:,int(17*cols/20):] = 0

	#feature binning with bins of size 20x20
	H,W,n = img.shape
	print(H,W)
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
        #feature tracking
	trackPts = cv2.KeyPoint_convert(kp)
	trackPts = np.expand_dims(trackPts, axis=1)
	lk_params = dict( winSize  = (15,15),maxLevel = 3,\
                      criteria = (cv2.TERM_CRITERIA_EPS | \
                      cv2.TERM_CRITERIA_COUNT, 50, 0.03))
	Pts_2, st, err = cv2.calcOpticalFlowPyrLK(\
                     prevl,imgl, trackPts, None,\
                     flags=cv2.MOTION_AFFINE, **lk_params)

	# separate points that were tracked successfully
	ptTrackable = np.where(st == 1, 1,0).astype(bool)
	TrkPts_1 = trackPts[ptTrackable, ...]
	TrkPts_2 = Pts_2[ptTrackable, ...]
	TrkPts_2 = np.around(TrkPts_2)

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
	#print ("Points with error less than " \
        #       + str(error) + " : " + str(len(TrkPts_1)))
	Pts_3DA = []
	Pts_3DB = []
	Pts_2DA = []
	Pts_2DB = []
	for j in range(len(TrkPts_1)):
		PtA = TrkPts_1[j]
		PtB = TrkPts_2[j]
		#print(PtA)
		#print(PtB)
		if int(PtA[1]) >= H or int(PtA[0]) >= W:
			continue
		else:
			dA = prev[int(PtA[1])][int(PtA[0])]/scale
		if int(PtB[1]) >= H or int(PtB[0]) >= W:
			continue
		else:
			dB = img[int(PtB[1])][int(PtB[0])]/scale


		#print(dA)
		#print(dB)
		dA=int(dA[0])
		dB=int(dB[0])
		if dA > 0 and dB > 0:
			Pts_3DA.append([base*(PtA[0] - cx)/dA,\
			                base*(PtA[1] - cy)/dA, f*base/dA])
			Pts_3DB.append([base*(PtB[0] - cx)/dB,\
			                base*(PtB[1] - cy)/dB, f*base/dB])
			Pts_2DA.append(PtA)
			Pts_2DB.append(PtB)
	Pts_2DA=np.asarray(Pts_2DA)
	Pts_2DB=np.asarray(Pts_2DB)
	Pts_3DA=np.asarray(Pts_3DA)
	Pts_3DB=np.asarray(Pts_3DB)

	
	# Outlier rejection
	if select_type=="OR":
		Compare3D = np.zeros((len(Pts_3DA),len(Pts_3DA)))
	
		for i in range(len(Pts_3DA)):
			for j in range(len(Pts_3DA)):
				Dis_1 = distance.euclidean(Pts_3DA[i],Pts_3DA[j])
				Dis_2 = distance.euclidean(Pts_3DB[i],Pts_3DB[j])
				Compare3D[i,j] = abs(Dis_1-Dis_2)

		Sum3D = np.sum(Compare3D,axis = 1)
		FinalIndex = np.argsort(Sum3D)
	
	
		while len(Sum3D) > 10:
			Compare3D = np.delete(Compare3D,FinalIndex[len(Sum3D)-1],0)
			Compare3D = np.delete(Compare3D,FinalIndex[len(Sum3D)-1],1)
			Pts_2DA = np.delete(Pts_2DA,FinalIndex[len(Sum3D)-1],0)
			Pts_2DB = np.delete(Pts_2DB,FinalIndex[len(Sum3D)-1],0)
			Pts_3DA = np.delete(Pts_3DA,FinalIndex[len(Sum3D)-1],0)
			Pts_3DB = np.delete(Pts_3DB,FinalIndex[len(Sum3D)-1],0)
			Sum3D = np.sum(Compare3D,axis = 1)
			FinalIndex = np.argsort(Sum3D)
		#homogenisation - append 1 as w
		homo = np.ones((len(Pts_3DA),1))
		Pts_1F = np.hstack((Pts_2DA,homo))
		Pts_2F = np.hstack((Pts_2DB,homo))
		Pts3D_1F = np.hstack((Pts_3DA,homo))
		Pts3D_2F = np.hstack((Pts_3DB,homo))

	
		#print(Pts3D_2F)
	# Inlier detection
	else:
		clique = find_bestPts_ID(Pts_3DA,Pts_3DB,10)
	
		Pts_1F = [Pts_2DA[i] for i in clique]
		Pts_2F = [Pts_2DB[i] for i in clique]
		Pts3D_1F = [Pts_3DA[i] for i in clique]
		Pts3D_2F = [Pts_3DB[i] for i in clique]
	
		#print(np.asarray(Pts3D_1F))
		homo = np.ones((len(Pts3D_1F),1))
		Pts_1F = np.hstack((Pts_1F,homo))
		Pts_2F = np.hstack((Pts_2F,homo))
		Pts3D_1F = np.hstack((Pts3D_1F,homo))
		Pts3D_2F = np.hstack((Pts3D_2F,homo))
	dSeed = np.zeros(len(Pts3D_1F))
	optRes = least_squares(mini, dSeed, method='lm', \
	max_nfev=200,args=(Pts3D_1F, Pts3D_2F, Pts_1F,Pts_2F,P1_roi))
	# Finding Rotation and Translation
	Rmat = genEulerZXZMatrix(\
	optRes.x[0], optRes.x[1], optRes.x[2])
	Trans = np.array(\
	[[optRes.x[3]], [optRes.x[4]], [optRes.x[5]]])
	CurGT[0] = float(GT_file_lines[ID].split()[4])
	CurGT[1] = float(GT_file_lines[ID].split()[5])
	# Updating the odometry
	newPosition = np.vstack(\
	(np.hstack((Rmat,Trans)),[0, 0, 0, 1]))
	Position = np.dot(Position,newPosition)
	# Processing Ground Truth for plotting
	GT_x = CurGT[0] - StartGT[0]
	GT_y = CurGT[1] - StartGT[1]
	
	# Resizing map dynamically
	while (centre+int(GT_x) >= Map.shape[0]-25 or\
		centre-int(Position[0,3]) >= Map.shape[0]-25 or\
                centre-int(GT_y) >= Map.shape[1]-25 or \
                int(Position[2,3])+centre >= Map.shape[1]-25):
		Map = np.insert(Map,len(Map[0]),0,axis=1)
		Map = np.insert(Map,len(Map),0,axis=0)
	while (centre+int(GT_x) <= 25 or\
                centre-int(Position[0,3]) <= 25 or\
                centre-int(GT_y) <= 25 or\
		int(Position[2,3])+centre <= 25):
		Map = np.insert(Map,0,0,axis=1)
		Map = np.insert(Map,0,0,axis=0)
		centre+=1
	# Plotting Ground Truth point in RED
	cv2.circle(Map,(centre+int(GT_x),centre-int(GT_y)),\
                    2,color=(0,0,255),thickness = -2)
	
	# Plotting VO point in Green
	cv2.circle(Map, (centre-int(Position[0,3]),\
                    int(Position[2,3])+centre),2,\
                    color=(0,255,0), thickness = -1)
	
	#disp_bgr = cv2.cvtColor(disp_view,cv2.COLOR_GRAY2BGR)
	mask_bgr = cv2.cvtColor(img_mask,cv2.COLOR_GRAY2BGR)

	disp_bgr = cv2.resize(disp_view,None,fx=0.5, fy=0.5,\
                           interpolation = cv2.INTER_CUBIC)
	mask_bgr = cv2.resize(mask_bgr,None,fx=0.5, fy=0.5,\
                           interpolation = cv2.INTER_CUBIC)

	display_image = np.concatenate((disp_bgr,mask_bgr),1)
	if display_image.shape[1]>=disp_bgr.shape[1]:
		display_image = np.concatenate(\
              (disp_bgr,display_image[:,0:disp_bgr.shape[1],:]),0)
	else:
		display_image = np.concatenate(\
              (disp_bgr[:,0:display_image.shape[1],:], display_image),0)

	display_image = cv2.resize(display_image,None,\
                    fx=0.75, fy=0.75,interpolation = cv2.INTER_CUBIC)
	scale1 = float(display_image.shape[0])/float(Map.shape[0])
	display_map = cv2.resize(Map,None,fx=scale1,fy=scale1,\
                                interpolation = cv2.INTER_CUBIC)
	display = np.concatenate((display_image, display_map),1)
	print(display.shape)

	if ID <107:
		out.write(display)
	elif ID==107:
		out.release()
		print("done!")
		
	prev = img
	prevl = imgl
	ID= ID + 1
	print(ID)
	k = cv2.waitKey(2)

	if k == ord('p'):
		Pause = not Pause
	elif k == 27:
		break

