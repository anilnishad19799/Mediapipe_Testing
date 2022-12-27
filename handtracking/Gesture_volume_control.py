import cv2
from cv2 import VariationalRefinement
import numpy as np
import mediapipe as mp
import time
import handtracking_module as htm
import math
# import pycaw
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

###################################
# Params

# 0. CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
# 1. CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
# 2. CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file
# 3. CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
# 4. CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
# 5. CV_CAP_PROP_FPS Frame rate.
# 6. CV_CAP_PROP_FOURCC 4-character code of codec.
# 7. CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
# 8. CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
# 9. CV_CAP_PROP_MODE Backend-specific value indicating the current capture mode.
# 10. CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
# 11. CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).
# 12. CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).
# 13. CV_CAP_PROP_HUE Hue of the image (only for cameras).
# 14. CV_CAP_PROP_GAIN Gain of the image (only for cameras).
# 15. CV_CAP_PROP_EXPOSURE Exposure (only for cameras).
# 16. CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
# 17. CV_CAP_PROP_WHITE_BALANCE Currently unsupported
# 18. CV_CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)

###################################
wCam, hCam = 640,480
##################################



cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

pTime = 0

detector = htm.handDetector()

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
volBar = 400
vol = 0
volPer = 0
while True:
    success, img = cap.read()
    img = detector.findHands(img)

    lmlist = detector.findPosition(img,draw=False)
    # print(lmlist)

    if len(lmlist)!=0:
        # print(lmlist[4], lmlist[8])

        x1,y1 = lmlist[4][1], lmlist[4][2]
        x2,y2 = lmlist[8][1], lmlist[8][2]
        cx,cy = (x1+x2) // 2, (y1+y2) // 2

        cv2.circle(img,(x1,y1),15,(0,0,255),cv2.FILLED)
        cv2.circle(img,(x2,y2),15,(0,0,255),cv2.FILLED)
        cv2.circle(img,(cx,cy),15,(0,0,255),cv2.FILLED)
        cv2.line(img , (x1,y1),(x2,y2),(0,0,255),2)

        length = math.hypot(x2-x1, y2-y1)
        # print(length)

        # Hand finger range 50-300
        # volume range -65 - 0

        vol = np.interp(length,[50,300],[minVol,maxVol])
        volBar = np.interp(length,[50,300],[400,150])
        volPer = np.interp(length,[50,300],[0,100])


        print(vol)
        volume.SetMasterVolumeLevel(vol, None)


        if length < 50:
            cv2.circle(img,(cx,cy),15,(0,255,0),cv2.FILLED)
    
    cv2.rectangle(img,(50,150),(85,400),(0,255,0),3)
    cv2.rectangle(img,(50,int(volBar)),(85,400),(0,255,0),cv2.FILLED)
    cv2.putText(img,f"{int(volPer)}%", (40,450), cv2.FONT_HERSHEY_COMPLEX, 1 , (255,0,0), 2)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,f"FPS:{int(fps)}", (40,50), cv2.FONT_HERSHEY_COMPLEX, 1 , (255,0,0), 2)

    cv2.imshow('img',img)
    cv2.waitKey(1)