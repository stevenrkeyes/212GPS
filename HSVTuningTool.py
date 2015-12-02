### HSV Tuning tool

import cv2
import numpy as np

def nothing(x):
    pass

c1= cv2.VideoCapture(0)

# todo: implement this, then resize the window
# set the camera to be wide aspect ratio
#c1.set(cv2.CAP_PROP_FRAME_HEIGHT,900) #1080
#c1.set(cv2.CAP_PROP_FRAME_WIDTH,1440) #1920

_,img = c1.read()
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)

# create trackbars for color change
cv2.createTrackbar('H UP','image',0,255,nothing)
cv2.createTrackbar('H LOW','image',0,255,nothing)
cv2.createTrackbar('S UP','image',0,255,nothing)
cv2.createTrackbar('S LOW','image',0,255,nothing)
cv2.createTrackbar('V UP','image',0,255,nothing)
cv2.createTrackbar('V LOW','image',0,255,nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'image',0,1,nothing)

# set default trackbar positions
cv2.setTrackbarPos('H UP','image',255)
cv2.setTrackbarPos('S UP','image',255)
cv2.setTrackbarPos('V UP','image',255)
cv2.setTrackbarPos(switch,'image',1)

while(1):
    _,img = c1.read()
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

    # get current positions of four trackbars
    H_U = cv2.getTrackbarPos('H UP','image')
    H_L = cv2.getTrackbarPos('H LOW','image')
    S_U = cv2.getTrackbarPos('S UP','image')
    S_L = cv2.getTrackbarPos('S LOW','image')
    V_U = cv2.getTrackbarPos('V UP','image')
    V_L = cv2.getTrackbarPos('V LOW','image')
    s = cv2.getTrackbarPos(switch,'image')
    H_UFliter = np.array([H_U, 255, 255])
    H_LFliter = np.array([H_L, 0, 0])
    S_UFliter = np.array([255, S_U, 255])
    S_LFliter = np.array([0, S_L, 0])
    V_UFliter = np.array([255, 255, V_U])
    V_LFliter = np.array([0, 0, V_L])

    upperFilter = np.array([255,255,255])
    lowerFilter = np.array([0,0,0])
    if s == 0:
        cv2.imshow('image', img)
    else:
    	if H_L > H_U:
    		maskH = cv2.bitwise_not(cv2.inRange(hsv, np.array([H_U, 0, 0]), np.array([H_L, 255, 255])))
        else:
        	maskH = cv2.inRange(hsv, H_LFliter, H_UFliter)
        if S_L > S_U:
    		maskS = cv2.bitwise_not(cv2.inRange(hsv, np.array([0, S_U, 0]), np.array([255, S_L, 255])))
        else:
        	maskS = cv2.inRange(hsv, S_LFliter, S_UFliter)
        if V_L > V_U:
    		maskV = cv2.bitwise_not(cv2.inRange(hsv, np.array([0, 0, V_U]), np.array([255, 255, V_L])))
        else:
        	maskV = cv2.inRange(hsv, V_LFliter, V_UFliter)
        maskHS = cv2.bitwise_and(maskS,maskS, mask= maskH)
        maskHSV = cv2.bitwise_and(maskHS, maskHS, mask=maskV)


        img2 = cv2.bitwise_and(img,img, mask= maskHSV)
        cv2.putText(img2,'Filter: ['+str(H_U)+','+str(S_U)+','+str(V_U)+'] , [' +str(H_L)+','+str(S_L)+','+str(V_L)+']', (20,20),
                    cv2.FONT_HERSHEY_SIMPLEX, .625, (0,0,255),1 )

        cv2.imshow('image', img2)
cv2.destroyAllWindows()
