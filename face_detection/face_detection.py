import cv2
import mediapipe as mp
import time


myFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = myFaceDetection.FaceDetection(0.75)

cap = cv2.VideoCapture(0)

pTime=0

while True:
    success,img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = faceDetection.process(imgRGB)

    if result.detections:
        for id,detection in enumerate(result.detections):
            # Inbuilt function for detection face
            # mpDraw.draw_detection(img,detection)

            # print(id,detection)

            # get detection score
            # print(detection.score)

            # this will give you x,y,w,h coordinate of bounding boxes
            # print(detection.location_data.relative_bounding_box)

            # my made function for detection bounding box
            bboxC = detection.location_data.relative_bounding_box
            ih,iw,ic = img.shape

            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img,bbox,(255,0,255),2)
            cv2.putText(img,f'{int(detection.score[0]*100)}%', (bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_PLAIN, 2,(255,0,255),3)



    # print(result)
    cTime = time.time()
    fps = (1/(cTime - pTime))
    pTime = cTime

    cv2.putText(img,str(int(fps)), (70,50),cv2.FONT_HERSHEY_PLAIN, 3,(255,0,255),3)
        

    cv2.imshow('img',img)
    cv2.waitKey(1)