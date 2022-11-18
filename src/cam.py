import numpy as np
import cv2
import tkinter as tk
from tkinter import *
from submain import *
import os
from PIL import Image, ImageTk


def CamPage():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_alt2.xml')

    cancel = False
    cap = cv2.VideoCapture(0)
    bgColor = "#fffde7"

    # Set default image
    dirname = os.path.dirname(__file__)
    pathname = dirname.replace("src", "")

    while(True):
        global camImage, imagePath1, imagePath2
        # Capture per frame
        ret, frame = cap.read()    

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        displayCam = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            img_item = "./test/camInput/imageCam.png"
            cv2.imwrite(img_item, roi_gray)

            # To show rectangle detection of faces
            color = (255,0,0) # BGR 
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame,(x,y), (end_cord_x, end_cord_y), color, stroke)


        # Display the resulting frame
        cv2.imshow('', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    
