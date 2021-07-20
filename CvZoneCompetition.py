import cv2
import numpy as np
from time import sleep
import random

length_min = 80 # Minimum length of retangle
height_min = 80 # Minimum height of the angle

offset = 6 #Error allowed between pixel

pos_linha = 550 

delay = 60 #FPS of video

detect = []
cars = 0


def paste_center (x, y, w, h):
    x1 = int (w / 2)
    y1 = int (h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

cap = cv2.VideoCapture ("DRONE-SURVEILLANCE-CONTEST-VIDEO.mp4")
cap.set (3,500)
cap.set (4,500)
subtractor = cv2.bgsegm.createBackgroundSubtractorMOG ()

while True:
    ret, frame1 = cap.read ()
    time = float(1 / delay)
    sleep(time)
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 10)
    img_sub = subtractor.apply(blur)
    dilate = cv2.dilate(img_sub, np.ones ((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.morphologyEx(dilate, cv2. MORPH_CLOSE, kernel)
    dilated = cv2.morphologyEx(dilated, cv2. MORPH_CLOSE, kernel)
    contour, h = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame1, (25, pos_linha), (1900, pos_linha), (255,0,0), 3)
    for (i, c) in enumerate(contour):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_contour = (w >= length_min) and (h >= height_min)
        if not validate_contour:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0,255,0), 2)
        center = paste_center (x, y, w, h)
        detect.append(center)
        cv2.circle(frame1, center, 4, (0, 0.255), -1)
        cv2.putText(frame1,str(random.randint(1,200)),(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2) 
        for (x, y) in detect:
            if y <(pos_linha + offset) and y> (pos_linha-offset):
                cars += 1
                cv2.line(frame1, (25, pos_linha), (1200, pos_linha), (0,127,255), 3)
                cv2.putText(frame1, str (random.randint (1,200)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                detect.remove((x, y))
                print("car is detected:" + str (cars))
    cv2.putText(frame1, "Moran 11", (850, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.putText(frame1, str(cars), (1700, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Surveillance Video", frame1)
    

    if cv2.waitKey (10) == 27:
        break
    
cv2.destroyAllWindows ()
cap.release ()
