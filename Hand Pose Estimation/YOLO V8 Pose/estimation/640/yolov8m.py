from ultralytics import YOLO
import math
import cv2
import time
import numpy as np
import tkinter
import pyautogui as pg
import os

dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())))

model_path = dir + "\\models\\yolo 640\\yolov8m\\best_float32.tflite"
tflite_model = YOLO(model_path,task='pose')

classnames = ['hand']

smoothening = 7
ploc_x,ploc_y = 0, 0
fr = 0
wcam, hcam = 640, 480
pTIme = 0
cTime = 0

root = tkinter.Tk()
root.withdraw()
WIDTH, HEIGHT = root.winfo_screenwidth(), root.winfo_screenheight()
print(WIDTH, HEIGHT)
pTIme = 0
cTime = 0

vid = os.path.join(os.path.dirname(os.path.join(os.path.dirname(os.path.dirname(os.getcwd()))))) + '\\test_vids\\yohand3.mp4'
cap = cv2.VideoCapture(vid)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,wcam)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

pg.FAILSAFE = False

while True:
    success,img = cap.read()
    if success:
        results = tflite_model.track(img,persist=True)
        
        annotated_frame = results[0].plot()

    # if len(results)>0:
    #     for r in results[0]:
    #         boxes = r.boxes
    #         kps = r.keypoints
    #         thumbxy,indexxy = kps.xy[0]

    #         for box in boxes:
    #             x1,y1,x2,y2 = box.xyxy[0].tolist()
                
    #             # final_roi = [x1,y1,x2-x1,y2-y1]
    #             # print(final_roi)
    #             # ret = tracker1.init(img,final_roi)
    #             # success, roi = tracker1.update(img)
    #             # x,y,w,h = tuple(map(int,roi))
    #             # roii = img[y:y+h,x:x+w]
        
    #             # if success:
    #             #     p1 = (x,y)
    #             #     p2 = (x+w,y+h)
                
    #             x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
    #             p1 = (x1,y1)
    #             p2 = (x2,y2)
                
    #             conf = math.ceil((box.conf[0] * 100)) / 100

    #             clss = int(box.cls[0])

    #             if conf>0.5:
    #                 cv2.rectangle(img,p1,p2,(0,255,0),3)
    #                 print(str(classnames[clss]) + ' ' + str(conf))
    #                 cv2.putText(img, f'{str(classnames[clss])}  {str(conf)}', (max(0,x1), max(35,y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    
    #                 tconf, iconf = kps.conf[0]
    #                 xCenter = (x1 + x2) / 2
    #                 yCenter = (y1 + y2) / 2
                
    #                 if tconf>0.5 and iconf>0.5:
    #                     img = cv2.circle(img, (int(thumbxy[0]), int(thumbxy[1])), 12, (0, 0, 0), -10) ##thumb
    #                     img = cv2.circle(img, (int(indexxy[0]), int(indexxy[1])), 12, (0, 255, 0), -10) ##index


                        ## code related to mouse stuff
                        # length = math.hypot(indexxy[0] - thumbxy[0], indexxy[1] - thumbxy[1])

                        # if length<=30:
                        # cd_dif = f'{abs(length)}'
                        # # cv2.putText(img,cd_dif,(10,70),cv2.FONT_HERSHEY_PLAIN,3,(0,0,0),3)
                        # print(cd_dif)
                        # m,n = indexxy[0],indexxy[1]
                
                        # x3 = np.interp(m,(fr,wcam-fr),(0,WIDTH))
                        # y3 = np.interp(n,(fr,hcam-fr),(0,HEIGHT))

                        # cloc_x = ploc_x + (x3-ploc_x) / smoothening
                        # cloc_y = ploc_y + (y3-ploc_y) / smoothening
                    
                        # pg.moveTo(WIDTH-cloc_x,cloc_y)
                        # ploc_x,ploc_y = cloc_x,cloc_y
                
                        # if length <=70:
                        #     print('click')
                        #     # pg.click(_pause=False,clicks=1,interval=1)"""

        cTime = time.time()
        fps = 1 / (cTime-pTIme)
        pTIme = cTime
        cv2.putText(annotated_frame,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

        cv2.imshow("YOLO v8m",annotated_frame)
        k = cv2.waitKey(1) & 0xff

        if k==27:
            break
    
    else:
        break

cap.release()
cv2.destroyAllWindows()