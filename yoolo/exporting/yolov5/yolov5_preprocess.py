# coding: utf-8
import torch
import cv2
import numpy as np
import math
import time
from . import kneron_preprocessing
kneron_preprocessing.API.set_default_as_520()
torch.backends.cudnn.deterministic = True
img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']
def make_divisible(x, divisor):
    # Returns x evenly divisble by divisor
    return math.ceil(x / divisor) * divisor

def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size

def letterbox_ori(img, new_shape=(640, 640), color=(0, 0, 0), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r)) # width, height 
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        #img = kneron_preprocessing.API.resize(img,size=new_unpad, keep_ratio = False)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # top, bottom = int(0), int(round(dh + 0.1))
    # left, right = int(0), int(round(dw + 0.1))    
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    #img = kneron_preprocessing.API.pad(img, left, right, top, bottom, 0)

    return img, ratio, (dw, dh)

def letterbox(img, new_shape=(640, 640), color=(0, 0, 0), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r)) # width, height 
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    # dw /= 2  # divide padding into 2 sides
    # dh /= 2

    if shape[::-1] != new_unpad:  # resize
        #img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        img = kneron_preprocessing.API.resize(img,size=new_unpad, keep_ratio = False)

    # top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    # left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    top, bottom = int(0), int(round(dh + 0.1))
    left, right = int(0), int(round(dw + 0.1))    
    #img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    img = kneron_preprocessing.API.pad(img, left, right, top, bottom, 0)

    return img, ratio, (dw, dh)

def letterbox_test(img, new_shape=(640, 640), color=(0, 0, 0), auto=True, scaleFill=False, scaleup=True):

    ratio = 1.0, 1.0
    dw, dh = 0, 0
    img = kneron_preprocessing.API.resize(img, size=(480, 256), keep_ratio=False, type='bilinear')
    return img, ratio, (dw, dh)

def LoadImages(path,img_size):  #_rgb # for inference
    if isinstance(path, str):
        img0 = cv2.imread(path)  # BGR       
    else:
        img0 = path  # BGR

    # Padded resize
    img = letterbox(img0, new_shape=img_size)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    return img, img0

def LoadImages_yyy(path,img_size): #_yyy # for inference
    if isinstance(path, str):
        img0 = cv2.imread(path)  # BGR       
    else:
        img0 = path  # BGR

    yvu = cv2.cvtColor(img0, cv2.COLOR_BGR2YCrCb)
    y, v, u = cv2.split(yvu)
    img0 = np.stack((y,)*3, axis=-1)

    # Padded resize
    img = letterbox(img0, new_shape=img_size)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    return img, img0

def LoadImages_yuv420(path,img_size):  #_yuv420 # for inference 
    if isinstance(path, str):
        img0 = cv2.imread(path)  # BGR       
    else:
        img0 = path  # BGR
    img_h, img_w = img0.shape[:2]
    img_h = (img_h // 2) * 2
    img_w = (img_w // 2) * 2
    img = img0[:img_h,:img_w,:]
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
    img0= cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420) #yuv420

    
    # Padded resize
    img = letterbox(img0, new_shape=img_size)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    return img, img0

def Yolov5_preprocess(image_path, device, imgsz_h, imgsz_w) : 
    model_stride_max = 32
    imgsz_h = check_img_size(imgsz_h, s=model_stride_max)  # check img_size
    imgsz_w = check_img_size(imgsz_w, s=model_stride_max)  # check img_size
    img, im0 = LoadImages(image_path, img_size=(imgsz_h,imgsz_w))
    img = kneron_preprocessing.API.norm(img) #path1
    #print('img',img.shape)
    img = torch.from_numpy(img).to(device) #path1,path2
    # img = img.float()  # uint8 to fp16/32 #path2
    # img /= 255.0#256.0 - 0.5 # 0 - 255 to -0.5 - 0.5 #path2
    
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    return img, im0

