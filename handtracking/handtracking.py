## Importing library
import enum
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import time

mpHands = mp.solutions.hands
#### there are four parameter in this Hands() function 
# 1. static_image   --> default is False 
# 2. number_of_hand  --> default is 2
# 3. min_detection_confidence   --> default is 0.5
# 4. min_tracking_confidence    --> default is 0.5
# only uses RGB images
hands = mpHands.Hands()    

# to draw a line import libraries
mpDraw = mp.solutions.drawing_utils 

cap = cv2.VideoCapture(0)

cTime = 0
pTime = 0

while True:
    success , img = cap.read()

    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # This will process the image
    result = hands.process(imgRGB)
    # this is used to detect the landmark 
    # print(result.multi_hand_landmarks)

    # to draw a line between each point of hand and detect point of landmark
    # mpHands.HAND_CONNECTIONS is used for joining the point of landmark
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            # For getting info of points of landmarks
            for id,lm in enumerate(handLms.landmark):
                # Here we are getting id of landmark and x,y,z coordinates in which the landmarks is in decimal points like x = 0.667 which is ratio of height  
                # then if we do multiply by height and width then we get pixel value
                # print(id,lm)
                h,w,ch = img.shape
                cx,cy = int(lm.x*w) , int(lm.y*h)
                print(id,cx,cy)

                # Put some condition for getting target landmark

                if id==4:
                    cv2.circle(img, (cx,cy), 15,(0,0,255),cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms ,mpHands.HAND_CONNECTIONS)

    # For finding the frame rate of video
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3 , (0,0,255), 2)

    cv2.imshow('img',img)
    cv2.waitKey(1)