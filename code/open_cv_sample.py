"""
    Getting started
    http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
    # check the frame width and height by cap.get(3) and cap.get(4)
    # But I want to modify it to 320x240. Just use ret = cap.set(3,320) and ret = cap.set(4,240).
"""

import numpy as np
import cv2

import os
import sys

APP_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
sys.path.append(APP_PATH)
DATA_PATH = os.path.join(APP_PATH, 'data')
IEMOCAP_PATH = os.path.join(DATA_PATH, 'IEMOCAP')

RUNNING_IN_DOCKER = True

if RUNNING_IN_DOCKER:
    IEMOCAP_PATH = '/app/IEMOCAP'

video_name = os.path.join(IEMOCAP_PATH, 'Ses01F_impro01.avi')
if not os.path.exists(video_name):
    print('File NOT found.\nExiting, adeus..')

print('Starting')
cap = cv2.VideoCapture(video_name)
while(cap.isOpened()):

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print('Finished')