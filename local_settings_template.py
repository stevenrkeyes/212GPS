# Note: copy this file to local_settings.py and adjust these settings

import numpy as np

########################################	Networking Setup
# ip = 'localhost'
ip = '192.168.1.121'

# Filter for the image by hsv into red and green
upperRed = np.array([210,255,250])
lowerRed = np.array([0,130,154])
upperGreen = np.array([85,95,176])
lowerGreen = np.array([27,10,120])

# see perspective_test.py for how i generated this
# todo: automate this
transformMatrix = np.array([[  0.917886537,     0.0454522979,    -236.261348],
                            [ -0.039397547,     0.821894407,      5.19774316],
                            [ -0.0000525885505, 0.0000236409881,  1.00000000]], np.float32)

