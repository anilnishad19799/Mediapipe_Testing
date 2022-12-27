import cv2
import mediapipe as mp
import time

############# THERE ARE TOTAL 468 POINTS IN FACEMESH #################### 

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()

# to draw the circle and lines thickness by slight modiication in mpDraw
drawSpec = mpDraw.DrawingSpec(thickness=1,circle_radius=1)

cap = cv2.VideoCapture(0)
pTime = 0
while True:

    success,img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS,
                                    drawSpec,drawSpec)

            for id,lm in enumerate(faceLms.landmark):
                # print(lm)
                ih,iw,ic = img.shape
                x,y = int(lm.x * iw) , int(lm.y * ih)
                print(id,x,y)


    cTime = time.time()
    fps = (1/(cTime - pTime))
    pTime = cTime

    cv2.putText(img,str(int(fps)), (20,70),cv2.FONT_HERSHEY_PLAIN, 3,(255,0,255),3)
        
    cv2.imshow('img',img)
    cv2.waitKey(1)