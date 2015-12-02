import numpy as np

########################################	Networking Setup
# ip = 'localhost'
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

