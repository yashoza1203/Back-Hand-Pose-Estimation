import cv2
import numpy as np
import tkinter
import time
from torch import nn
import timm
import torch
import os

MODEL_NAME = 'efficientnet_b0'

class ObjLocModel(nn.Module):
  def __init__(self):
    super(ObjLocModel,self).__init__()
    self.efficientnet = timm.create_model(MODEL_NAME, pretrained=True,num_classes=8)

  def forward(self, images, gt_bboxes=None,gt_kps=None):
    bboxes = self.efficientnet(images)

    if gt_bboxes != None:
      loss= nn.MSELoss()(bboxes, gt_bboxes)
      return bboxes,loss

    return bboxes

okay_now = False
smoothening = 2
ploc_x,ploc_y = 0, 0
fr = 0
wcam, hcam = 224, 224
pTIme = 0
cTime = 0
root = tkinter.Tk()
root.withdraw()
WIDTH, HEIGHT = root.winfo_screenwidth(), root.winfo_screenheight()

col = (255,0,0)
tcol = (0, 0, 255)
icol = (0, 255, 0)

model_p = os.getcwd() + "\\model\\EffNet_2.pt"
DEVICE = 'cpu'
vid = os.getcwd().rstrip('EfficientNet') + '\\test_vids\\yohand2.mp4'

cap = cv2.VideoCapture(vid)
model = ObjLocModel()
model.load_state_dict(torch.load(model_p,weights_only=True ,map_location=torch.device('cpu')))

def plot_bbox_kps(img,target,col,tcol,icol,tf=1):
  bbox_kps = list(target.cpu().numpy())
  bbox_kps = [bk.astype(int) for bk in bbox_kps][0]
#   print(bbox_kps)
  xmin = bbox_kps[0]
  ymin = bbox_kps[1]
  xmax = bbox_kps[2]
  ymax = bbox_kps[3]
  tx = bbox_kps[4]
  ty = bbox_kps[5]
  ix = bbox_kps[6]
  iy = bbox_kps[7]
  
  if tf==0:
    img = torch.squeeze(img,0)
    img = img.permute(1, 2, 0).numpy()

  img = cv2.rectangle(img, (xmin,ymin), (xmax,ymax), col, 1)
  img = cv2.circle(img, (tx,ty), 4, tcol, -10) ##thumb
  img = cv2.circle(img, (ix,iy), 4, icol, -10) ##index

  return img

while True:
    success,img = cap.read()
    if success:
        with torch.no_grad():
            img = cv2.resize(img,(224,224))
            image = torch.from_numpy(img).permute(2,0,1).unsqueeze(0) # (batch-size, c, h, w)
            out_box = model(image.float())
            img = plot_bbox_kps(img,out_box,col,tcol,icol)
    
            cTime = time.time()
            fps = 1 / (cTime-pTIme)
            pTIme = cTime
            cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(0,0,0),3)

            cv2.imshow("EfficientNet",img)
            k = cv2.waitKey(1) & 0xff

        if k==27:
          break
    else:
       break
        
cap.release()    
cv2.destroyAllWindows()