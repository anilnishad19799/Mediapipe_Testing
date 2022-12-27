import cv2
import mediapipe as mp
import time

class poseDetector():
    
    def __init__(self,mode=False,upBody =False, smooth=False, 
                detectionCon = 0.5,trackCon=0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self,img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.pose.process(imgRGB)

        if self.result.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.result.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def findposition(self,img,draw=True):
        lmlist = []
        for id,lm in enumerate(self.result.pose_landmarks.landmark):
            h,w,c = img.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            lmlist.append([id,cx,cy])
            # print(id,lm)
            if draw:
                cv2.circle(img, (cx,cy), 5, (255,0,0),cv2.FILLED)

        return lmlist

def main():
    
    cap = cv2.VideoCapture('E:/Anil/slow motion/VID-20200114-WA0046.mp4')

    pTime=0

    detector = poseDetector()
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


if __name__=="__main__":
    main()