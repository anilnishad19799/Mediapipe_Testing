import cv2
import mediapipe as mp
import time


mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture('E:/Anil/slow motion/VID-20200114-WA0046.mp4')

pTime=0
while True:

    success,img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(imgRGB)

    if result.pose_landmarks:
        mpDraw.draw_landmarks(img,result.pose_landmarks, mpPose.POSE_CONNECTIONS)

        for id,lm in enumerate(result.pose_landmarks.landmark):
            h,w,c = img.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            print(id,lm)
            cv2.circle(img, (cx,cy), 5, (255,0,0),cv2.FILLED)

    cTime = time.time()
    fps = (1/(cTime - pTime))
    pTime = cTime

    cv2.putText(img,str(int(fps)), (70,50),cv2.FONT_HERSHEY_PLAIN, 3,(0,255,0),3)
    
    cv2.imshow('image',img)
    cv2.waitKey(10)