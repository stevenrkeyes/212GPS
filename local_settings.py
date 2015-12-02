import numpy as np
import socket

# get the name of the computer to decide which local settings to use
computerName = socket.gethostname()

if computerName == 'yosaffbridge':
	########################################	Networking Setup
	ip = '192.168.1.212'

	# Filter for the image by hsv into red and green
	upperRed = np.array([185,255,220])
	lowerRed = np.array([155,120,50])
	upperGreen = np.array([80,130,150])
	lowerGreen = np.array([45,30,50])

	upperParking = np.array([255,255,255])
	lowerParking = np.array([0,38,0])

	# see perspective_test.py for how i generated this
	# todo: automate this
	transformMatrix = np.array([[  1.12492807,       -0.112526607,    -110.647370],
	                            [  0.0132346733,      0.984851257,    -17.4750276],
	                            [ -0.00000360866716, -0.000128203941,  1.00000000]], np.float32)

	parkingSpotX = 350
	parkingSpotY = 175
else: # hostname == 'abraxas'
	########################################	Networking Setup
    ip = '192.168.1.121'

    # Filter for the image by hsv into red and green
    upperRed = np.array([255,255,242])
    lowerRed = np.array([171,119,137])
    upperGreen = np.array([89,109,174])
    lowerGreen = np.array([36,13,111])

    upperParking = np.array([255,255,255])
    lowerParking = np.array([0,38,0])

    # see perspective_test.py for how i generated this
    # todo: automate this
    transformMatrix = np.array([[  0.917886537,     0.0454522979,    -236.261348],
                                [ -0.039397547,     0.821894407,      5.19774316],
                                [ -0.0000525885505, 0.0000236409881,  1.00000000]], np.float32)

    parkingSpotX = 300
    parkingSpotY = 175
