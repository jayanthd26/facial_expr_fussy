# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt


eye=cv2.imread('er1.jpg')

eyeycc = cv2.cvtColor(eye, cv2.COLOR_BGR2YCR_CB)
y,cb,cr=cv2.split(eyeycc)

eyecann = cv2.Canny(eye,100,250)
plt.imshow(eyecann)


lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")

frame=eye.copy()
converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
skinMask = cv2.inRange(converted, lower, upper)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
skinMask = cv2.erode(skinMask, kernel, iterations = 2)
skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

#skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
skin = cv2.bitwise_and(frame, frame, mask = skinMask)

skmask=cv2.bitwise_not(skinMask)

cv2.imshow("images", np.hstack([frame, skin]))

cv2.imshow('img',skmask)
cv2.waitKey(0)
cv2.destroyAllWindows()

