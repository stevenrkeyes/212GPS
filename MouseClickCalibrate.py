## mouse calibration tools

import cv2
import numpy as np
from time import sleep

c1 = cv2.VideoCapture(0)

img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('snapshot')
cv2.namedWindow('image')

buttonCount = 0
clickList = []
MaxPoints = 4

# mouse callback function
def getXY(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
    	global buttonCount
    	global clickList

    	buttonCount+=1
    	if buttonCount<MaxPoints:
    		clickList.append((x,y))
    		img2=img1[:].copy()
    		cv2.putText(img2, str(buttonCount) + '/'+str(MaxPoints), (20,20),
                    cv2.FONT_HERSHEY_SIMPLEX, .625, (0,0,255),1 )
    		cv2.putText(img2, str(clickList), (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, .625, (0,0,255),1 )
    		cv2.imshow('snapshot', img2)

        if buttonCount == MaxPoints:
        	clickList.append((x,y))
        	img2 = img1[:].copy()
	    	cv2.putText(img2, str(buttonCount) + '/'+str(MaxPoints)+' --> COMPLETE', (20,20),
                    cv2.FONT_HERSHEY_SIMPLEX, .625, (0,0,255),1 )
	    	cv2.putText(img2, str(clickList), (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, .625, (0,0,255),1 )
	    	cv2.imshow('snapshot', img2)
	    	clickList = []
	    	buttonCount = 0


cv2.setMouseCallback('snapshot',getXY)


while (1):
	_, img = c1.read()
	cv2.imshow('image', img)

	k = cv2.waitKey(1) & 0xFF
	if k == ord('q'):
		break
	if k== ord('s'):
		img1 = img[:]
		cv2.imshow('snapshot', img)
		clickList = []
	   	buttonCount = 0

    	

