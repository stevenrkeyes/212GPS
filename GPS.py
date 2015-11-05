"""
2.12 GPS Vision Server
Daniel J. Gonzalez - dgonz@mit.edu
Zack Bright - zbright@mit.edu
Fall 2015
"""

import threading,SocketServer,time
import signal
import sys
import struct
import numpy as np
import cv2
import math


########################################	Networking Setup
# ip = 'localhost'
ip = '192.168.1.212'
tStart = time.time()
timestamp = tStart
state = [0, 0, 0, 0, 0, 0]



class requestHandler(SocketServer.StreamRequestHandler):
    def handle(self):
        requestForUpdate=self.request.recv(256)
        print 'Connected to Client: ', str(self.client_address)
        while requestForUpdate!='':
            data1 = ''.join([struct.pack('>H',x) for x in state])
            data2 = struct.pack('>f',timestamp)
            self.wfile.write(data1+data2)
            requestForUpdate=self.request.recv(256)
        print('client disconnect')

class broadcastServer(SocketServer.ThreadingMixIn, SocketServer.TCPServer):
    pass

server=broadcastServer((ip,2121),requestHandler)
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

writeVid = 0

if writeVid:
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    filenameOUT = 'output_' + time.strftime("%Y%m%d-%H%M%S") + '.mov'
    out = cv2.VideoWriter(filenameOUT, fourcc, 20, (900,1440))
    val = out.open(filenameOUT, fourcc, 20, (900,1440))

########################################    SigInt Setup
def signal_handler(signal, frame):
    print('Closing...')
    server.socket.close()
    c.stop_capture()
    c.disconnect()
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

########################################    Main Loop

cv2.namedWindow('top', cv2.WINDOW_AUTOSIZE)

xOld = 0
yOld = 0
n = 0
vOld = 0
vxOld = 0
vyOld = 0
v=0
alph = (1/(1+0.016667))

tStart = time.time()
timestamp = time.time()+0.0001 - tStart
timestampOld = timestamp
maxTime = 25

while timestamp<maxTime:
    ####-----------------------------Top View Cam
    _,im = c1.read()
    img = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    cimg = im
    hsvimg = cv2.cvtColor(cimg,cv2.COLOR_BGR2HSV)

    # make black img of same size as hsvimg
    rimg = 255-np.zeros_like(hsvimg)
    gimg = 255-np.zeros_like(hsvimg)

    for x in range(len(hsvimg[0,:])):
        for y in range(len(hsvimg[:,0])):
            hue = hsvimg[y,x][0]
            if hue < .05 or hue >.95:
                rimg[y,x] = 0
            if hue < .40 and hue > .28:
                gimg[y,x] = 0

    cv2.imshow('red',rimg)
    cv2.imshow('green',gimg)
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,
        minDist = 2000,param1=350,param2=7,minRadius=70,maxRadius=100)# DO NOT CHANGE, WORKS PERFECTLY
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,
        minDist = 2000,param1=350,param2=7,minRadius=5,maxRadius=50) #zack edit of parameters

    if circles is None:
        [x, y, r] = [0, 0, 0]
    else:
        circles = np.uint16(np.around(circles))
        [x, y, r] = circles[0][0]
    
    ####-------------------------------------Draw
    if n%1==0:
        cv2.circle(cimg,(x,y),r,(0,255,0),2)
        cv2.circle(cimg,(x,y),2,(0,0,255),3)
        cv2.putText(cimg,'Velocity: '+str(int(v))+'  px/second', (20,20),
                    cv2.FONT_HERSHEY_SIMPLEX, .625, (50,50,0),1 )
       
        cv2.putText(cimg,str(int(n/timestamp))+' FPS', (550,20),
                    cv2.FONT_HERSHEY_SIMPLEX, .625, (50,0,0),1 )
        cv2.imshow('top',cimg)        
        if writeVid:
            out.write(cimg)
        ch = 0xFF & cv2.waitKey(5)
        if ch == ord('q'):
            break

    x2 = 0
    y2 = 0
    r2 = 0

    # update state
    state = (x, 472 - y, r*r*math.pi, x2, y2, r2*r2*math.pi)
    timestamp = time.time() - tStart
    dt = (timestamp-timestampOld)
    vx = (int(x)-xOld)/dt
    vx = alph*vxOld+(1-alph)*vx#First Order LPF
    vy = (int(y)-yOld)/dt
    vy = alph*vyOld+(1-alph)*vy#First Order LPF
    v = math.sqrt(vx*vx+vy*vy)
    xOld = x
    yOld = y
    timestampOld = timestamp
    vOld = v
    vxOld = vx
    vyOld = vy
    n+=1
print timestamp/60/60
print n/timestamp
c1.release()
print('Closing Server...')
server.socket.close()
if writeVid:
    out.release()