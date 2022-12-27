import cv2
import time
import pose_estimation_module as pm

pTime=0
while True:

    # cap = cv2.VideoCapture('E:/Anil/slow motion/VID-20200114-WA0046.mp4') ## for specific video
    cap = cv2.VideoCapture(0)  ## for webcam

    pTime=0

    detector = pm.poseDetector()
    while True:
        success,img = cap.read()  
        img = detector.findPose(img)
        lmlist = detector.findposition(img,draw=False)  
        if len(lmlist)!=0:
            # print(lmlist[14])
            cv2.circle(img, (lmlist[14][1],lmlist[14][2]), 15, (0,0,255),cv2.FILLED)
        cTime = time.time()
        fps = (1/(cTime - pTime))
        pTime = cTime

        cv2.putText(img,str(int(fps)), (70,50),cv2.FONT_HERSHEY_PLAIN, 3,(0,255,0),3)
        
        cv2.imshow('image',img)
        cv2.waitKey(10)
