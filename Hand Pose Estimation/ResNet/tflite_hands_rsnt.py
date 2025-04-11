import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import os

class Hands:
    def __init__(self,image=None,path=None):
        # self.model_path = 'C:/Users/hp/OneDrive/Desktop/virtual_mouse/cv_mouse1/model/mykerasmodel2.tflite'
        self.model_path = os.getcwd() + '//resnet_model.tflite'
        if path:
            self.path = path
        else:
            self.path=None
            self.image = image
 
    def get_image(self,image_path):
        img = Image.open(image_path)
        return img.resize((640,480))

    def preprocess_image(self,image_path=None,img=None):
        if image_path:
            img = Image.open(image_path)
        else:
            img = Image.fromarray(np.uint8(img))
        
        # img = img.convert('L').resize((640,480))
        img = img.resize((640,480))
        # print(img.size)
        img = tf.expand_dims(img, axis=0)
        # img_array = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=-1)
        img_array= np.array(img, dtype=np.float32) / 255.0
        # img_array = img_array[:, :, ::-1]

        return img_array

    def plot_bounding_boxes(self,image,pred_coords=[],pred_kps=[],norm=False):
        if norm:
            image *= 255
            image = image.astype('uint8')

        draw = ImageDraw.Draw(image)
        h,w = 480, 640

        xmin, ymin, xmax, ymax = pred_coords
        tx, ty, ix, iy = pred_kps

        xmin, ymin, xmax, ymax = int(xmin*w), int(ymin*h), int(xmax*w), int(ymax*h)

        draw.rectangle((xmin, ymin, xmax, ymax),outline='red',width = 3)
        draw.ellipse((int(tx*w),int(ty*h),int((tx*w)+10),int((ty*h)+10)), fill="green") ##THUMB
        draw.ellipse((int(ix*w), int(iy*h),int((ix*w)+10),int((iy*h)+10)), fill="blue") ##INDEX
        return image,[xmin, ymin, xmax, ymax]

    def get_bbox(self):
        interpreter = tf.lite.Interpreter(self.model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        if self.path:
            print('path to h')
            input_data = self.preprocess_image(image_path = self.path)
            img = self.get_image(self.path)
        else:
            input_data = self.preprocess_image(img=self.image)
            img = Image.fromarray(self.image).resize((640,480))

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        pred_kps = interpreter.get_tensor(output_details[0]['index'])
        pred_coords = interpreter.get_tensor(output_details[1]['index'])
        image,coords = self.plot_bounding_boxes(img,pred_coords[0],pred_kps[0],norm=False)

        return image,coords

# ipath=None
# # ipath = 'C:/Users/hp/OneDrive/Desktop/virtual_mouse/hands.jpg'
# # ipath = 'E:/virtual_mouse/cv_mouse1/spidey.jpg'
# ipath = 'E:/virtual_mouse/cv_mouse1/demo_img.jpg'
# handss = Hands(path=ipath)
# image,_ = handss.get_bbox()
# if ipath:
#     final_image = np.asarray(image)
#     final_image = cv2.cvtColor(final_image,cv2.COLOR_BGR2RGB)
#     cv2.imshow('image window', final_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()