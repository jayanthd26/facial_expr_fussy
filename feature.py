# -*- coding: utf-8 -*-

from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2

det = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

image = cv2.imread('f2.jpeg')
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = det(gray, 1)

for (i, rect) in enumerate(rects):
	shape = predict(gray, rect)
	shape = face_utils.shape_to_np(shape)

	for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
		print(name," ",i,"-",j)
		clone = image.copy()
		for (x, y) in shape[i:j]:
			cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

		(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
		roi = image[y:y + h, x:x + w]
		roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

		cv2.imshow("parts", roi)
		#cv2.imshow("Image", clone)
		cv2.waitKey(0)
	output = face_utils.visualize_facial_landmarks(image, shape)
	cv2.imshow("Image", output)
	cv2.waitKey(0)