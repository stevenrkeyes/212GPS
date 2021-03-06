import cv2
import numpy as np
import matplotlib.pyplot as plt

filename = 'testimage.png'
img = cv2.imread(filename)

cv2.namedWindow('test', flags=cv2.WINDOW_NORMAL)
cv2.imshow('test',img)
cv2.waitKey(0)

# objpoints represents the actual location of points in the world
# in whatever units we define.
# I could define it in meters, but I'll define it in pixels
# so that I end up transforming the picture from a 1280px by 720px
# to another 1280px by 720px image.
#objp = np.zeros((4*3,3), np.float32)
#objp[:,:2] = np.mgrid[0:1280+1:240,0:720+1:240].T.reshape(-1,2)
objp = np.zeros((8,3), np.float32)

# I selected these locations because they're easy to see
# visually
# but I would choose different locations
# (such as ones with fiducial markers perhaps)
# to make this process automated.
# (and the fiducials should be at the same height as those on the robot
# or else of variying heights, just FYI, and you should include that 
# height information in objpoints instead of just leaving it as 0)
cornerLocations = np.array([[0,0],
							[1280,0],
							[0,720],
							[1280,720]])
wallMidpoints = np.array([[640,0],
						  [0,360],
						  [1280,360],
						  [640,720]])
# todo: do this in 3D instead of just 2D
objp[:,:2] = np.concatenate((cornerLocations, wallMidpoints))
objpoints = [objp]

# imgpoints is the location of those same points, but in pixel coordinates in the calibration image
# (can be fractions of a pixel though)

#plt.imshow(img)
#plt.show()

imgpoints = [np.array([[[69.2,0.0]], # top left
					   [[1256.0,3.6]], # top right
					   [[137.8,713.7]], # bottom left
					   [[1178.8,706.8]], # bottom right
					   [[678.0,1.0]], # top middle
					   [[101.8,375.8]], # middle left
			 		   [[1218.3,370.6]], # middle right
					   [[671.2,710.2]]], np.float32)] # bottom middle

# todo: maybe lens correction

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[:2][::-1],None,None)

h,  w = img.shape[:2]
alpha = 1
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),alpha,(w,h))

# undistort the lens distortion
img_lenscorrected = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
#x,y,w,h = roi
#img_lenscorrected = img_lenscorrected[y:y+h, x:x+w]
cv2.imshow('result',img_lenscorrected)
cv2.waitKey(0)


