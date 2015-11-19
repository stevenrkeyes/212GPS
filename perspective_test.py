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
# and ideally at the heights of the actual markers
cornerLocations = np.array([[0,0],
							[1280,0],
							[0,720],
							[1280,720]], np.float32)

# imgpoints is the location of those same points, but in pixel coordinates in the calibration image
# (can be fractions of a pixel though)

#plt.imshow(img)
#plt.show()

imgpoints = np.array([[69.2,0.0], # top left
					  [1256.0,3.6], # top right
					  [137.8,713.7], # bottom left
					  [1178.8,706.8], # bottom right
					  ], np.float32)


h,  w = img.shape[:2]

transformMatrix = cv2.getPerspectiveTransform(imgpoints, cornerLocations)
print transformMatrix
size = img.shape[:2][::-1]
img_transformed = cv2.warpPerspective(img, transformMatrix, size)
cv2.imshow('result', img_transformed)
cv2.waitKey(0)