import os
import cv2
import time
import numpy as np
import imgaug as ia
from iou import iou
from imgaug import augmenters as iaa
from ultralytics import YOLO

model_path_224 =  "E:\\virtual_mouse\\final models mouse\\models\\yolo 224\\yolov8n\\best_float32.tflite"
model_path_224_n =  "E:\\virtual_mouse\\final models mouse\\models\\yolo 224\\yolov8n\\best.onnx"
model_path_224_s =  "E:\\virtual_mouse\\final models mouse\\models\\yolo 224\\yolov8s\\best.onnx"
model_path_224_m =  "E:\\virtual_mouse\\final models mouse\\models\\yolo 224\\yolov8m\\best.onnx"
model_path_224_x =  "E:\\virtual_mouse\\final models mouse\\models\\yolo 224\\yolov8x\\best.onnx"

model_path_640_x =  "E:\\virtual_mouse\\final models mouse\\models\\yolo 640\\yolov8x\\best.onnx"

tflite_model = YOLO(model_path_224,task='pose')
metrics = tflite_model.val()
print(metrics)