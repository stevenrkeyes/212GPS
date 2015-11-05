import numpy as np
import cv2

if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    while True:
        _,im = cam.read()
        #im2 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Preview', im)
        ch = 0xFF & cv2.waitKey(5)