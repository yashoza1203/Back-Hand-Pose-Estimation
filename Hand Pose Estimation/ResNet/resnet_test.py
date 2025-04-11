import cv2
import numpy as np
import tkinter
import time
from tflite_hands_rsnt import Hands
import os

okay_now = False
smoothening = 2
ploc_x,ploc_y = 0, 0
fr = 0
wcam, hcam = 480, 640
pTIme = 0
cTime = 0
root = tkinter.Tk()
root.withdraw()
WIDTH, HEIGHT = root.winfo_screenwidth(), root.winfo_screenheight()

# vid = "E:/virtual_mouse/yohand2.mp4"
# cap = cv2.VideoCapture(vid)

DEVICE = 'cpu'
vid = os.getcwd().rstrip('ResNet') + '\\test_vids\\yohand3.mp4'

cap = cv2.VideoCapture(vid)

while True:
    success,img = cap.read()

    if success:
        img = cv2.resize(img,(480,640))
        hanss = Hands(image=img)
        handp,coords = hanss.get_bbox()

        # roi = detect_hand(rooi)
        # final_roi = [roi[0],roi[1],roi[2]-roi[0],roi[3]-roi[1]]
        # final_roi = [x1,y1,x2-x1,y2-y1]
        handp = np.array(handp)
        cTime = time.time()
        fps = 1 / (cTime-pTIme)
        pTIme = cTime
        cv2.putText(handp,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(0,0,0),3)

        cv2.imshow("Resnet",handp)
        k = cv2.waitKey(1) & 0xff

        if k==27:
            break
        
cap.release()    
cv2.destroyAllWindows()