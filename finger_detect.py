import cv2
import numpy as np
import copy
import math

# Open Camera
camera = cv2.VideoCapture(0)
img = np.zeros((1024,1024,3), np.uint8)
draw=True
while True:
    ret, frame = camera.read()
    frame = cv2.resize(frame, (1024,1024))
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # Smoothing
    frame = cv2.flip(frame, 1)  #Horizontal Flip
    cv2.imshow('original', frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20,255, 255], dtype="uint8")
    skinMask = cv2.inRange(hsv, lower, upper)
    cv2.imshow('Threshold Hands', skinMask)

    # Getting the contours and convex hull
    skinMask1 = copy.deepcopy(skinMask)
    contours, hierarchy = cv2.findContours(skinMask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    maxArea = -1
    if length > 0:
        for i in range(length):
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > maxArea:
                maxArea = area
                ci = i

        res = contours[ci]
        hull = cv2.convexHull(res)

        M = cv2.moments(res)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

    c = max(contours, key=cv2.contourArea)
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    
    drawing = np.zeros(frame.shape, np.uint8)

    cv2.drawContours(drawing, [c], -1, (0, 255, 255), 2)
    cv2.circle(drawing, extLeft, 8, (0, 0, 255), -1)
    cv2.circle(drawing, extRight, 8, (0, 255, 0), -1)
    cv2.circle(drawing, extTop, 8, (100,55,100), -1)
    cv2.circle(frame, extTop, 8, (100,55,100), -1)
    cv2.circle(drawing, extBot, 8, (255, 255, 0), -1)
    
    cv2.drawContours(frame, [res], 0, (0, 255, 0), 2)
    cv2.drawContours(frame, [hull], 0, (0, 0, 255), 3)
    cv2.circle(drawing, (cX, cY), 7, (255, 255, 255), -1)
    cv2.putText(drawing, "center", (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # cv2.imshow('output', drawing)

    edges = cv2.Canny(img,100,140)

    cv2.imshow('original', frame)
    cv2.imshow("write",img)

    if draw == True:
        cv2.circle(img, extTop, 10, (100,55,100), -1)
        cv2.circle(img, (extTop[0],extTop[1]+10), 10, (0,55,100), -1)
    
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
        break    
    elif k == ord('n'):
        img = np.zeros((1024,1024,3), np.uint8)
    if k == 9:
        draw=True if draw==False else False
