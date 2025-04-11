from tflite_hands_rsnt import Hands
import math
import cv2
import time
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

def detect_hand(img):
    hand_img = img.copy()
    hanss = Hands(image=hand_img)
    h,coords = hanss.get_bbox()
    h.save('pred.jpg')
    return coords

classnames = ['hand']
pTIme = 0
cTime = 0

while True:
    ret, frame = cap.read()

    roi = detect_hand(frame)

    cv2.rectangle(frame,(roi[0],roi[1]),(roi[2],roi[3]),(0,0,0))

    cTime = time.time()
    fps = 1 / (cTime-pTIme)
    pTIme = cTime

    cv2.putText(frame,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(0,0,0),3)

    cv2.imshow('Resu',frame)
    k = cv2.waitKey(1) & 0xff

    if k==27:
        break

cap.release()    
cv2.destroyAllWindows()