## mouse calibration tools

import cv2
import numpy as np
from time import sleep

#c1 = cv2.VideoCapture(0)
c1 = cv2.VideoCapture('testimage.png')

#c1.set(cv2.CAP_PROP_FRAME_HEIGHT,900) #1080
#c1.set(cv2.CAP_PROP_FRAME_WIDTH,1440) #1920

cv2.namedWindow('image')

buttonCount = 0
clickList = []
MaxPoints = 4

cornerList = ['top left','top right','bottom left','bottom right']

# I selected these locations because they're easy to see
cornerLocations = np.array([[0,0],
			    [1280,0],
			    [0,720],
			    [1280,720]], np.float32)

def printTransformMatrix():
	global clickList
	imgpoints = np.array(clickList, np.float32)
	transformMatrix = cv2.getPerspectiveTransform(imgpoints, cornerLocations)

	np.set_printoptions(suppress=True)
	# why doesn't numpy have a function for this, like matlab's mat2str D:
	print 'transformMatrix = ' + 'np.array(' + str([list(row) for row in transformMatrix]) + ', np.' + str(np.result_type(transformMatrix)) + ')'
	
	size = img.shape[:2][::-1]
	img_transformed = cv2.warpPerspective(img, transformMatrix, size)
	
	cv2.imshow('image', img_transformed)
	cv2.waitKey(0)


# mouse callback function
def getXY(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
    	global buttonCount
    	global clickList

    	buttonCount+=1
    	if buttonCount<MaxPoints:
    		clickList.append((x,y))
    		cv2.putText(img, str(buttonCount) + '/'+str(MaxPoints), (20,20),
                    cv2.FONT_HERSHEY_SIMPLEX, .625, (0,0,255),1 )
    		cv2.putText(img, str(clickList), (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, .625, (0,0,255),1 )
    		cv2.imshow('image', img)

        if buttonCount == MaxPoints:
        	clickList.append((x,y))
	    	cv2.putText(img, str(buttonCount) + '/'+str(MaxPoints)+' --> COMPLETE', (20,20),
                    cv2.FONT_HERSHEY_SIMPLEX, .625, (0,0,255),1 )
	    	cv2.putText(img, str(clickList), (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, .625, (0,0,255),1 )
	    	cv2.imshow('image', img)
		print ''
		print 'Transformation matrix calculated. Please paste the following into local settings:'
		printTransformMatrix()
	    	clickList = []
	    	buttonCount = 0


cv2.setMouseCallback('image',getXY)

_, img = c1.read()
cv2.putText(img, 'Please click the ' + cornerList[buttonCount] + 'corner.', (100,100),
                    cv2.FONT_HERSHEY_SIMPLEX, .625, (0,0,255),1 )
cv2.imshow('image', img)

while (1):
#	_, img = c1.read()
	cv2.putText(img, 'Please click the ' + cornerList[buttonCount] + 'corner.', (100,100),
                    cv2.FONT_HERSHEY_SIMPLEX, .625, (0,0,255),1 )
	cv2.imshow('image', img)

	k = cv2.waitKey(1) & 0xFF
	if k == ord('q'):
		break
