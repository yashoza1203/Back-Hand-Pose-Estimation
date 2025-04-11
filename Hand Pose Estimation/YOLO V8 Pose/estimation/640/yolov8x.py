from ultralytics import YOLO
import cv2
import time
import tkinter
import pyautogui as pg
import os

dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())))

model_path = dir + "\\models\\yolo 640\\yolov8x\\best_float32.tflite"
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
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,hcam)

pg.FAILSAFE = False

while True:
    success,img = cap.read()
    if success:
        results = tflite_model.track(img,persist=True)
        
        annotated_frame = results[0].plot()

        cTime = time.time()
        fps = 1 / (cTime-pTIme)
        pTIme = cTime
        cv2.putText(annotated_frame,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

        cv2.imshow("YOLO v8x",annotated_frame)
        k = cv2.waitKey(1) & 0xff

        if k==27:
            break
    
    else:
        break

cap.release()
cv2.destroyAllWindows()