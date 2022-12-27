## Importing library
import enum
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import time



class handDetector():
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        # print(self.mode,self.maxHands,self.detectionCon,self.trackCon)
        self.hands = self.mpHands.Hands()    # here parameter like mode,maxHands etx is dicaraded because error is shon
        self.mpDraw = mp.solutions.drawing_utils 


    def findHands(self,img, draw = True):

        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)
        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms ,self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self,img,handNo=0, draw = True):   

        lmlist = []

        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):
                h,w,ch = img.shape
                cx,cy = int(lm.x*w) , int(lm.y*h)
                # print(id,cx,cy)
                lmlist.append([id,cx,cy])

                # if id==4:
                if draw:
                    cv2.circle(img, (cx,cy), 15,(0,0,255),cv2.FILLED)

        return lmlist

def main():
    cap = cv2.VideoCapture(0)
    cTime = 0
    pTime = 0
    detector = handDetector()

    while True:
        success , img = cap.read()
        img = detector.findHands(img,draw=True)
        lmlist = detector.findPosition(img)
        if len(lmlist) != 0:
            print(lmlist[4])

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3 , (0,0,255), 2)

        cv2.imshow('img',img)
        cv2.waitKey(1)




if __name__ == '__main__':
    main()