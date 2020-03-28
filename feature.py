# -*- coding: utf-8 -*-

from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import faceFussyValues as ffv
import pandas as pd
det = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def get_fussy_values(img_path):
    image = cv2.imread(img_path)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    rects = det(gray, 1)
    
    for (i,rect) in enumerate(rects):
        shape=predict(gray,rect)
        shape=face_utils.shape_to_np(shape)        
        width_eye=ffv.weidth_of_eye(shape[36][0],shape[39][0],shape[42][0],shape[45][0])
        he1=ffv.hight_of_eyebrow_1(shape[17][1],shape[26][1],shape[33][1])
        he2=ffv.hight_of_eyebrow_1(shape[21][1],shape[22][1],shape[33][1])
        wm=ffv.width_mouth(shape[48][0],shape[54][0])
        om=ffv.opening_mouth(shape[62][1],shape[66][1])
        nl=ffv.nose_tiplip(shape[33][1],shape[48][1],shape[54][1])
        #ec=ffv.eye_check()
        '''
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            print(name+" (",i,",",j,")")
            clone=image.copy()
            for (x,y) in shape[i:j]:
                cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
            (x,y,w,h)=cv2.boundingRect(np.array([shape[i:j]]))
            roi=image[y:y + h, x:x + w]
            roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
            cv2.imshow("parts", roi)
            #cv2.imshow("Image", clone)
            cv2.waitKey(0)
            cv2.destroyAllWindows()'''
        return width_eye,he1,he2,wm,om,nl

import os

angry=[]

directory = r'CK+48/anger'
for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        print(os.path.join(directory, filename))
        angry.append(list(get_fussy_values(os.path.join(directory, filename))))
        
    else:
        print("112132")
 
    
    
sad=[]

directory = r'CK+48/sadness'
for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        print(os.path.join(directory, filename))
        sad.append(list(get_fussy_values(os.path.join(directory, filename))))
        
    else:
        print("112132")


happy=[]

directory = r'CK+48/happy'
i=0
for filename in os.listdir(directory):
    i=i+1
    if filename.endswith(".jpg") or filename.endswith(".png"):
        print(os.path.join(directory, filename))
        try:
            happy.append(list(get_fussy_values(os.path.join(directory, filename))))
        except:
            print(filename)
        
    else:
        print("112132")


angry=np.array(angry)
sad=np.array(sad)
happy=np.array(happy)


angry_df = pd.DataFrame({'ew': angry[:, 0], 'he1': angry[:, 1], 'he2': angry[:, 2], 'wm': angry[:, 3], 'om': angry[:, 4], 'nl': angry[:, 5]})
sad_df = pd.DataFrame({'ew': sad[:, 0], 'he1': sad[:, 1], 'he2': sad[:, 2], 'wm': sad[:, 3], 'om': sad[:, 4], 'nl': sad[:, 5]})
happy_df = pd.DataFrame({'ew': happy[:, 0], 'he1': happy[:, 1], 'he2': happy[:, 2], 'wm': happy[:, 3], 'om': happy[:, 4], 'nl': happy[:, 5]})


angry_df.to_csv ('angry.csv', index = False, header=True)
sad_df.to_csv('sad.csv',index=False,header=True)
happy_df.to_csv('happy.csv',index=False,header=True)