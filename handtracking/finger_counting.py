import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import time
import os 
import handtracking_module as htm

wCam, hCam = 640,480

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

folderPath = 'G:/Computer_vision/Advance_computer_vision_mediapose/Images/'
myList = os.listdir(folderPath)
# print(myList)
overlayList = []

for imgPath in myList:
    image = cv2.imread(f'{folderPath}/{imgPath}')
    overlayList.append(image)

print(len(overlayList))

cTime = 0
pTime = 0

detector = htm.handDetector()

tipIds = [4,8,12,16,20]

while True:
    success , img = cap.read()

    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)
    # print(lmlist)

    if len(lmlist)!=0:
        finger = []

        # thumb
        if lmlist[tipIds[0]][1] > lmlist[tipIds[0]-1][1]:
                finger.append(1)
        else:
            finger.append(0)

        # 4 finger
        for id in range(1,5):
            if lmlist[tipIds[id]][2] < lmlist[tipIds[id]-2][2]:
                finger.append(1)
            else:
                finger.append(0)

        # print(finger)

        totalFingers = finger.count(1)
        print(totalFingers)

    # h,w,c = overlayList[totalFingers-1].shape

    # img[0:h, 0:w] = overlayList[totalFingers-1]

        cv2.rectangle(img,(20,255),(170,425),(0,255,0),cv2.FILLED)
        cv2.putText(img, str(totalFingers),(43,375),cv2.FONT_HERSHEY_PLAIN,10,(255,0,0),25)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,f'FPS :{str(int(fps))}', (400,70), cv2.FONT_HERSHEY_PLAIN, 3 , (0,0,255), 2)

    cv2.imshow('img',img)
    cv2.waitKey(1)