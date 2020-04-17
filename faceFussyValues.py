# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
def weidth_of_eye(x5,x6,x10,x11):
    return ((x11-x10)+(x6-x5))/2


def hight_of_eyebrow_1(y2,y4,y15):
    return ((y15-y4)+(y15-y2))/2


def hight_of_eyebrow_2(y1,y3,y15):
    return ((y15-y3)+(y15-y1))/2  


def width_mouth(x18,x19):
    return x19-x18


def opening_mouth(y20,y21):
    return y21-y20

def nose_tiplip(y15,y18,y19):
    return ((y18-y15)+(y19-y15))/2

def eye_check(y9,y14,y16,y17):
    return ((y16-y9)+(y17-y14))/ 2


def get_irises_location(frame_gray):
    eye_cascade = cv2.CascadeClassifier( 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(frame_gray, 1.3, 10)  # if not empty - eyes detected
    irises = []

    for (ex, ey, ew, eh) in eyes:
        iris_w = int(ex + float(ew / 2))
        iris_h = int(ey + float(eh / 2))
        irises.append([np.float32(iris_w), np.float32(iris_h)])

    return np.array(irises)