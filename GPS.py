"""
2.12 GPS Vision Server
Daniel J. Gonzalez - dgonz@mit.edu
Zack Bright - zbright@mit.edu
Steven Keyes - srkeyes@mit.edu
Fall 2015
"""

import threading,SocketServer,time
import signal
import sys
import struct
import numpy as np
import cv2
import math

import broadcast_server

# import machine-specific settings for camera calibration, etc
from local_settings import *

tStart = time.time()
timestamp = tStart
state = [0, 0, 0]



class requestHandler(SocketServer.StreamRequestHandler):
    def handle(self):
        requestForUpdate=self.request.recv(256)
        print 'Connected to Client: ', str(self.client_address)
        while requestForUpdate!='':
            data1 = ''.join([struct.pack('>f',x) for x in state])
            data2 = struct.pack('>f',timestamp)
            self.wfile.write(data1+data2)
            requestForUpdate=self.request.recv(256)
        print('client disconnect')

server=broadcast_server.broadcastServer((ip,2121),requestHandler)
t = threading.Thread(target=server.serve_forever)
t.daemon=True
t.start()
print('server start')


########################################	Connect To Cameras
c1= cv2.VideoCapture(0)

print c1.get(cv2.CAP_PROP_FRAME_HEIGHT)
print c1.get(cv2.CAP_PROP_FRAME_WIDTH)
c1.set(cv2.CAP_PROP_FRAME_HEIGHT,900) #1080
c1.set(cv2.CAP_PROP_FRAME_WIDTH,1440) #1920
print c1.get(cv2.CAP_PROP_FRAME_HEIGHT)
print c1.get(cv2.CAP_PROP_FRAME_WIDTH)

print "Connected to Camera 1." 
"""
c2= cv2.VideoCapture(1)
print "Connected to Camera 2."  
"""

########################################    Setup Video Save

writeVid = False

if writeVid:
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    filenameOUT = 'output_' + time.strftime("%Y%m%d-%H%M%S") + '.mov'
    out = cv2.VideoWriter(filenameOUT, fourcc, 20, (900,1440))
    val = out.open(filenameOUT, fourcc, 20, (900,1440))

########################################    SigInt Setup
def signal_handler(signal, frame):
    print('Closing...')
    server.socket.close()
    server.disconnect()
    c1.release()
    if writeVid:
        out.release()
    c.stop_capture()
    c.disconnect()
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

########################################    Main Loop

def distanceFun(coordinatePair):
    # todo: use something faster than python ints but that won't overflow
    # like numpy int32 or something
    deltaX = int(coordinatePair[0][0]) - int(coordinatePair[1][0])
    deltaY = int(coordinatePair[0][1]) - int(coordinatePair[1][1])
    return np.hypot(deltaX, deltaY)

cv2.namedWindow('top', cv2.WINDOW_AUTOSIZE)

xOld = 0
yOld = 0
n = 0
alph = (1/(1+0.016667))

tStart = time.time()
timestamp = time.time()+0.0001 - tStart
timestampOld = timestamp
maxTime = 120000

# initialize these values to something for a fallback
# in case no circles are detected (in which case the
# previous values are used)
# and as an initial value for the low pass filter
# (note: the result is that the first few values are
# skewed to the (0,0,0) point)
# todo: wait to initialize these until actual circles
# are found for the first time
stateInitialized = False
[x, y, phi] = [0, 0, 0]
[xg, yg, rg] = [0, 0, 0]
[xr, yr, rr] = [0, 0, 0]

METERS_PER_PIXEL = 2.0/720

# parameters for low pass filter
# note: sampling period is estimated, not enforced
# todo: enforce sampling period or recalculate it
samplingPeriod = 0.037 #seconds
cutoffFrequency = 1 #Hz
alpha = 2 * math.pi * samplingPeriod * cutoffFrequency / (1 + 2 * math.pi * samplingPeriod * cutoffFrequency)

size = (1280,720)

while timestamp<maxTime:
    ####-----------------------------Top View Cam
    _,im = c1.read()
    # Perspective correction
    im = cv2.warpPerspective(im, transformMatrix, size)
    img = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    cimg = im
    hsvimg = cv2.cvtColor(cimg,cv2.COLOR_BGR2HSV)

    # todo: correct for lens distortion (different from perspective correction)

    #gmask = cv2.inRange(hsvimg, lowerGreen, upperGreen)
    #rmask = cv2.inRange(hsvimg, lowerRed, upperRed)

    # close the image a bit
    #gmask = cv2.erode(gmask, np.ones((2,2),np.uint8))
    #gmask = cv2.dilate(gmask, np.ones((3,3),np.uint8))

    #gimg = cv2.bitwise_and(img,img,mask = gmask)
    #rimg = rmask*img

    # find the best circle candidates using the hsv image
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,
        minDist = 20,param1=350,param2=7,minRadius=9,maxRadius=13)

    if circles is None:
        # return [0,0,0], indicating an error
        # todo: actually, it would be better to do something else, such as [-1,-1,-1] or just return the last datapoint instead 
        pass # keep all the values the same as the previous iteration
    else:
        # extract the coordinates and radii from the circles found by hough transform
        circles = np.uint16(np.around(circles))

        # go through the circles until a green circle is found and a red circle is found
        # (in case there are other circles on the board, like a circular clump of snow)
        # (hopefully, this will be after just two iterations)
        redCircleFound = False
        greenCircleFound = False
        # note: I'm limiting it to 3 circles
        # todo: limit the number of circles to check to a small number, but do it in a way that won't
        # throw an out of range error if no circles are found
        possibleRedCircles = []
        possibleGreenCircles = []
        for circle in circles[0][0:20]:
            [xCircle, yCircle, rCircle] = circle
            
            # uncomment this line to visualize all detected circles
            #cv2.circle(cimg,(xCircle,yCircle),5,(0,255,255),1)

            # create a mask of the circle to check the pixels of the original image
            #circleMask = np.zeros_like(img)
            #cv2.circle(circleMask,(x0,y0),r0,(255,255,255),-1)

            # use the mask to extract the pixels from the original image
            #gimg = cv2.bitwise_and(img,circleMask)

            # figure out which of these circles is the red circle and which is the grn circle
            # todo: do something smarter than just checking the center pixel; for example, average of all pixels
            centerPixel = hsvimg[yCircle:yCircle+1, xCircle:xCircle+1]
            if cv2.inRange(centerPixel, lowerGreen, upperGreen):
                possibleGreenCircles.append([xCircle, yCircle, rCircle])
                #[xg, yg, rg] = [xCircle, yCircle, rCircle]
            elif cv2.inRange(centerPixel, botLowerRed, botUpperRed) or \
                 cv2.inRange(centerPixel, topLowerRed, topUpperRed):
                possibleRedCircles.append([xCircle, yCircle, rCircle])
                #[xr, yr, rr] = [xCircle, yCircle, rCircle]
        
        # evaluate which pair of circles makes the best candidate for the marker
        # based on how close the distance between them is to the expected distance
        possiblePairs = [[gCoords, rCoords] for gCoords in possibleGreenCircles for rCoords in possibleRedCircles]
            
        # only look at pairs that are within a resonable closeness to each other
        possiblePairs = [c for c in possiblePairs if (45 < distanceFun(c) < 75)]
        
        # find the pair whose distance is closest to the expected
        if len(possiblePairs) > 0:
            closestPair = min(possiblePairs, key=lambda c: abs(distanceFun(c) - 60))
            [xg, yg, rg] = closestPair[0]
            [xr, yr, rr] = closestPair[1]

        # calculate x, y, and phi METERS_PER_PIXEL
        xNew = METERS_PER_PIXEL*(0.5*(xg + xr) - 1280*0.5)
        yNew = METERS_PER_PIXEL*-(0.5*(yg + yr) - 720*0.5)
        # todo: using int to avoid overflow errors; what is a better way to do this?
        phiNew = -math.atan2(int(xg) - int(xr), -(int(yg) - int(yr)))

        # filter the values unless this is the first value detected
        if stateInitialized == False:
            x = xNew
            y = yNew
            phi = phiNew
            xg_filtered = xg
            yg_filtered = yg
            xr_filtered = xr
            yr_filtered = yr
            stateInitialized = True
        else:
            # low pass filter implemented as exponentially weighted moving average
            # filteredValue = alpha*unfilteredValue + (1-alpha)*prevFilteredValue
            x = alpha * xNew + (1 - alpha) * x
            y = alpha * yNew + (1 - alpha) * y
            xg_filtered = alpha * xg + (1 - alpha) * xg_filtered
            xr_filtered = alpha * xr + (1 - alpha) * xr_filtered
            yg_filtered = alpha * yg + (1 - alpha) * yg_filtered
            yr_filtered = alpha * yr + (1 - alpha) * yr_filtered
            # phi can't be filtered directly due to wrap-around
            phi = -math.atan2(int(xg_filtered) - int(xr_filtered), -(int(yg_filtered) - int(yr_filtered)))

        # todo: implement a scheme (such as with energy or velocity limits) to filter out outlier measurements
    
    # calculate the parking spot snow coverage
    parkingSpotImg = hsvimg[parkingSpotY:parkingSpotY+200,parkingSpotX:parkingSpotX+75]
    #cv2.imshow('park', parkingSpotImg)
    clearPixels = cv2.inRange(parkingSpotImg, lowerParking, upperParking)
    cv2.imshow('park2', clearPixels)
    # the highest possible sum, if all the parking spot and car are clear
    #print np.sum(clearPixels)
    totalPossiblePixels = 2050000
    fractionClear = np.sum(clearPixels) * 1.0 / totalPossiblePixels
    # round and limit to 0-100%
    percentClear = min(100, round(fractionClear, 1)*100)
    
    ####-------------------------------------Draw
    if n%1==0:
        # plot the cirlces found (unfiltered)

        # todo: handle the case that no circles are found or that x,y,phi are not initialized
        # (as would be the case if no circles have been found for the first few frames, like 
        # before a robot has been placed on the field)

        # green circle
        cv2.circle(cimg,(xg,yg),rg,(0,255,0),2)
        cv2.circle(cimg,(xg,yg),1,(0,0,0),3)

        # red circle
        cv2.circle(cimg,(xr,yr),rr,(0,0,255),2)
        cv2.circle(cimg,(xr,yr),1,(0,0,0),3)

        # robot heading
        pntGreen = np.array([xg,yg], np.int32)
        pntRed = np.array([xr,yr], np.int32)
        pntRobot = np.array([x/METERS_PER_PIXEL + 1280*0.5,
                             -y/METERS_PER_PIXEL + 720*0.5], np.float32)
        vectorFwd = np.array([math.cos(phi), -math.sin(phi)], np.float32)
        vectorLeft = np.array([-math.sin(phi), -math.cos(phi)], np.float32)
        d = pntGreen - pntRed
        dPerp = np.array([-d[1], d[0]], np.int32)
        points = np.array([pntRed*.75 + pntGreen*.25,
                          pntRed*.25 + pntGreen*.75,
                          pntRed*.25 + pntGreen*.75 + 0.5*dPerp,
                          pntRed*.5 + pntGreen*.5 + 0.7*dPerp,
                          pntRed*.75 + pntGreen*.25 + 0.5*dPerp],np.int32)
        points = np.array([pntRobot + 15*vectorLeft,
                           pntRobot + 15*vectorLeft + 30*vectorFwd,
                           pntRobot + 40*vectorFwd,
                           pntRobot - 15*vectorLeft + 30*vectorFwd,
                           pntRobot - 15*vectorLeft], np.int32)
        points.reshape((-1,1,2))
        cv2.fillConvexPoly(cimg,points,(34,139,34))
        
        cv2.putText(cimg,str(int(n/timestamp))+' FPS', (550,20),
                    cv2.FONT_HERSHEY_SIMPLEX, .625, (50,0,0),1 )
                    
        # indicate what amount of the parking spot is clear
        cv2.putText(cimg, "Percent Parking Spot Clear: " + str(percentClear) + "%", (50,20), cv2.FONT_HERSHEY_SIMPLEX, .625, (50,0,0),1 )
        
        cv2.imshow('top',cimg)        
        if writeVid:
            out.write(cimg)
        ch = 0xFF & cv2.waitKey(5)
        if ch == ord('q'):
            break

    # update state
    state = (x, y, phi)
    timestamp = time.time() - tStart
    dt = (timestamp-timestampOld)
    xOld = x
    yOld = y
    timestampOld = timestamp
    n+=1
        
print timestamp/60/60
print n/timestamp
c1.release()
print('Closing Server...')
server.socket.close()
if writeVid:
    out.release()
