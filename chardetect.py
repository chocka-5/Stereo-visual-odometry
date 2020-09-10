import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

file='/home/vidyaa/AP03TC9939.xml'

class Point: 
    def __init__(self, x, y): 
        self.x = x 
        self.y = y 
  
# Given three collinear points p, q, r, the function checks if  
# point q lies on line segment 'pr'  
def onSegment(p, q, r): 
    if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and 
           (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))): 
        return True
    return False
  
def orientation(p, q, r): 
    # to find the orientation of an ordered triplet (p,q,r) 
    # function returns the following values: 
    # 0 : Colinear points 
    # 1 : Clockwise points 
    # 2 : Counterclockwise       
    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y)) 
    if (val > 0): 
          
        # Clockwise orientation 
        return 1
    elif (val < 0): 
          
        # Counterclockwise orientation 
        return 2
    else: 
          
        # Colinear orientation 
        return 0
  
# The main function that returns true if  
# the line segment 'p1q1' and 'p2q2' intersect. 
def doIntersect(p1,q1,p2,q2): 
      
    # Find the 4 orientations required for  
    # the general and special cases 
    o1 = orientation(p1, q1, p2) 
    o2 = orientation(p1, q1, q2) 
    o3 = orientation(p2, q2, p1) 
    o4 = orientation(p2, q2, q1) 
  
    # General case 
    if ((o1 != o2) and (o3 != o4)): 
        return True
  
    # If none of the cases 
    return False


tree = ET.parse(file)
a=[]
root = tree.getroot()
for i in range(6,len(root)):
	for item in root[i]:
		if(item.tag == 'name'):
			a.append(item.text)	
		for subitem in item:
			a.append(subitem.text)

arr = np.reshape(a, (len(root)-6, 5)) 
#list of characters with corresponding parameters
arr =sorted(arr, key = lambda x: int(x[1]))
arr=np.array(arr) 
#calculating the mid points coordinates of first and last characters respectively
fmx=(int(arr[0][1])+int(arr[0][1]))/2
fmy=(int(arr[0][2])+int(arr[0][4]))/2
lmx=(int(arr[len(arr)-1][1])+int(arr[len(arr)-1][1]))/2
lmy=(int(arr[len(arr)-1][2])+int(arr[len(arr)-1][4]))/2
intersec=0
lpi=[]
lp=[]
finallp=""
#segregate the character into respective lines based on the intersection
for k in range(0, len(arr)):
	p1 = Point(fmx, fmy) 
	q1 = Point(lmx, lmy) 
	p2 = Point(int(arr[k][1]),int(arr[k][2])) 
	q2 = Point(int(arr[k][1]),int(arr[k][4])) 
	if doIntersect(p1, q1, p2, q2): 
	    intersec=intersec+1
	    lpi.append(k)
	else:
	    lp.append(k)
#...change the variable name for the two lines...
 
if(len(arr)>intersec):
	for p in range(0,len(lp)):
		pos=int(lp[p])
		finallp=finallp+arr[pos][0]
	for p in range(0,len(lpi)):
		pos=int(lpi[p])
		finallp=finallp+arr[pos][0]	
else:
	for p in range(0,len(arr)):
		finallp=finallp+arr[p][0]

print(finallp)		




