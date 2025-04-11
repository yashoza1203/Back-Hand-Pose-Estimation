import os
import cv2
import time
import numpy as np
import imgaug as ia
from iou import iou
from imgaug import augmenters as iaa
from ultralytics import YOLO
import os

dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())))

model_path_224 =  dir + "\\models\\yolo 224\\yolov8n\\best_float32.tflite"
model_path_224_n = dir + "\\models\\yolo 224\\yolov8n\\best.onnx"
model_path_224_s = dir + "\\models\\yolo 224\\yolov8s\\best.onnx"
model_path_224_m = dir + "\\models\\yolo 224\\yolov8m\\best.onnx"
model_path_224_x = dir + "\\models\\yolo 224\\yolov8x\\best.onnx"
model_path_640_x = dir + "\\models\\yolo 640\\yolov8x\\best.onnx"

tflite_model = YOLO(model_path_224_n,task='pose')

resolution = 1


def flip_horizontal(img, keys):
    """ Flipping """
    aug = iaa.Sequential([iaa.Fliplr(1.0)])
    seq_det = aug.to_deterministic()
    keys = ia.KeypointsOnImage([ia.Keypoint(x=keys[0], y=keys[1]),
                                ia.Keypoint(x=keys[2], y=keys[3]),
                                ia.Keypoint(x=keys[4], y=keys[5]),
                                ia.Keypoint(x=keys[6], y=keys[7])], shape=img.shape)

    image_aug = seq_det.augment_images([img])[0]
    keys_aug = seq_det.augment_keypoints([keys])[0]
    k = keys_aug.keypoints
    keys_aug = [k[0].x, k[0].y, k[1].x, k[1].y, k[2].x, k[2].y, k[3].x, k[3].y]

    return image_aug, keys_aug

def plot_bbox_kps(img,target,col,tcol,icol):
    bbox_kps = target #list(target.cpu().numpy())
    bbox_kps = [int(bk) for bk in bbox_kps]
    xmin = bbox_kps[0]
    ymin = bbox_kps[1]
    xmax = bbox_kps[2]
    ymax = bbox_kps[3]
    tx = bbox_kps[4]
    ty = bbox_kps[5]
    ix = bbox_kps[6]
    iy = bbox_kps[7]

    img = cv2.rectangle(img, (xmin,ymin), (xmax,ymax), col, 1)
    img = cv2.circle(img, (tx,ty), 4, tcol, -10) ##thumb
    img = cv2.circle(img, (ix,iy), 4, icol, -10) ##index

    return img

def compare_plots(image,gt_box,out_box):
    print(gt_box)
    print(out_box)
    col = (0,255,0)
    tcol = (0,255,0)
    icol = (0,255,0)
    img = plot_bbox_kps(image,gt_box,col,tcol,icol)

    col = (0,0,255)
    tcol = (0,0,255)
    icol = (0, 0, 255)
    img = plot_bbox_kps(img,out_box,col,tcol,icol)

    cv2.imshow("Pred thumb",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  
image_directory = 'E:/virtual_mouse/TI1K-Dataset/TI1K-Dataset-master/test/'
label_directory = 'E:/virtual_mouse/TI1K-Dataset/TI1K-Dataset-master/annotation/'
image_files = os.listdir(image_directory)

""" Ground truth label file for TI1K dataset """
file = open(label_directory + 'label.txt')
lines = file.readlines()
file.close()

total_error = np.zeros([1, 4])
avg_hand_detect_time = 0
avg_fingertip_detect_time = 0
avg_time = 0
avg_iou = 0
count = 0
distance_error = []

height = int(480 * resolution)
width = int(640 * resolution)
c=0

for image_file in image_files:
    """ Generating ground truths labels """
    image = cv2.imread(image_directory + image_file)
    image = cv2.resize(image, (width, height))
    name = image_file[:-4]
    splits = name.split('_')
    gt = []

    if 'TI1K' in splits:
        """ TI1K Dataset """
        label = []
        for line in lines:
            line = line.strip().split()
            if image_file == line[0]:
                label = line[1:]
                break

        label = [float(i) for i in label]
        x1 = label[0] * width
        y1 = label[1] * height
        x2 = label[2] * width
        y2 = label[3] * height
        xt = label[4] * width
        yt = label[5] * height
        xi = label[6] * width
        yi = label[7] * height

        gt = [x1, y1, x2, y2, xt, yt, xi, yi]
        image_flip, gt_flip = flip_horizontal(image, np.asarray(gt))
        gt_box = [x1, y1, x2, y2]
        gt_box_flip = [gt_flip[2], gt_flip[1], gt_flip[0], gt_flip[3]]
        image_flip = image_flip.copy()

    tic1 = time.time()
    """ Predictions for the test images """
    image = cv2.imread(image_directory + image_file)
    image = cv2.resize(image, (width, height))
    results = tflite_model(image,imgsz=224,iou=0.5,conf=0.5,verbose=False)

    if len(results)>0:
        for r in results[0]:
            boxes = r.boxes
            kps = r.keypoints
            thumbxy,indexxy = kps.xy[0].tolist()

            for box in boxes:
                x1,y1,x2,y2 = box.xyxy[0].tolist()
                pr_box = [x1,y1,x2,y2]

                IOU = iou(boxA=pr_box, boxB=gt_box)

                if IOU < 0.5:
                    continue

                avg_iou = avg_iou + IOU

        tl = (x1, y1)
        br = (x2, y2)
        position = []
        pr = [tl[0], tl[1], br[0], br[1], thumbxy[0], thumbxy[1], indexxy[0], indexxy[1]]

        # compare_plots(image,gt,pr)

        # Calculating error for fingertips only
        gt = np.asarray(gt[4:])
        pr = np.asarray(pr[4:])
        abs_err = abs(gt - pr)
        total_error = total_error + abs_err
        D = np.sqrt((gt[0] - gt[2]) ** 2 + (gt[1] - gt[3]) ** 2)
        D_hat = np.sqrt((pr[0] - pr[2]) ** 2 + (pr[1] - pr[3]) ** 2)
        distance_error.append(abs(D - D_hat))
        count = count + 1
        print('Detected Image: {0}  IOU: {1:.4f}'.format(count, IOU))

    """ Predictions for the flipped test images """

    results = tflite_model(image_flip ,imgsz=224,iou=0.5,conf=0.5,verbose=False)
    # imgsz=640,iou=0.5,conf=0.5

    if len(results)>0:
        for r in results[0]:
            boxes = r.boxes
            kps = r.keypoints
            thumbxy,indexxy = kps.xy[0].tolist()

            for box in boxes:
                x1,y1,x2,y2 = box.xyxy[0].tolist()
                pr_box = [x1,y1,x2,y2]

                IOU = iou(boxA=pr_box, boxB=gt_box)

                if IOU < 0.5:
                    continue
        
                avg_iou = avg_iou + IOU

        tl = (x1, y1)
        br = (x2, y2)
        position = []
        pr = [tl[0], tl[1], br[0], br[1], thumbxy[0], thumbxy[1], indexxy[0], indexxy[1]]

        # compare_plots(image_flip,gt_flip,pr)
        # Calculating error for fingertips only

        gt_flip = np.asarray(gt_flip[4:])
        pr = np.asarray(pr[4:])
        abs_err = abs(gt_flip - pr)
        total_error = total_error + abs_err
        D = np.sqrt((gt_flip[0] - gt_flip[2]) ** 2 + (gt_flip[1] - gt_flip[3]) ** 2)
        D_hat = np.sqrt((pr[0] - pr[2]) ** 2 + (pr[1] - pr[3]) ** 2)
        distance_error.append(abs(D - D_hat))
        count = count + 1
        print('Detected Image: {0}  IOU: {1:.4f}'.format(count, IOU))
        # if c>1:
        # c+=1
        toc1 = time.time()
        avg_time = avg_time + (toc1 - tic1)
        # break

er = total_error / count
avg_iou = avg_iou / count
er = er[0]
er = np.round(er, 4)
distance_error = np.array(distance_error)
distance_error = np.mean(distance_error)
distance_error = np.round(distance_error, 4)

avg_time = avg_time / 1000
print('Average execution time: {0:1.5f} ms'.format(avg_time * 1000))
print('Total Detected Image: {0}'.format(count))
print('Average IOU: {0}'.format(avg_iou))
print('Pixel errors: xt = {0}, yt = {1}, xi = {2}, yi = {3}, D-D_hat = {4}'.format(er[0], er[1],
                                                                                   er[2], er[3],
                                                                                   distance_error))

print('{0} & {1} & {2} & {3} & {4}'.format(er[0], er[1], er[2], er[3], distance_error))