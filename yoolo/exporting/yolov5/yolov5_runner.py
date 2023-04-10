import torch
torch.set_printoptions(precision=10)
torch.set_printoptions(threshold=99999999999)
torch.backends.cudnn.deterministic = True
from .yolov5_preprocess import *
from .yolov5_postprocess import *
from .yolo_v2 import Model as Model_v2

import time
import os
from collections import Counter
import torch.nn.functional as F
import random
from pathlib import Path

class Yolov5Runner:
    def __init__(self, model_path, yaml_path, grid20_path, grid40_path, grid80_path, num_classes, imgsz_h, imgsz_w, conf_thres, iou_thres, top_k_num, vanish_point, **kwargs):#is_onnx,
        """
        inputs :
            model_path : str ,path to model 
        """
        self.model_path = model_path
        self.imgsz_h = imgsz_h
        self.imgsz_w = imgsz_w
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.top_k_num = top_k_num
        self.vanish_point = vanish_point
        self.num_classes = num_classes
        self.DEVICE = torch.device("cpu")#torch.device('cuda:0')#
        self.grid20 = torch.from_numpy(np.load(grid20_path))
        self.grid40 = torch.from_numpy(np.load(grid40_path))
        self.grid80 = torch.from_numpy(np.load(grid80_path)) 
        self.grids = [self.grid80, self.grid40, self.grid20]        
        if 'onnx' not in model_path:
            #if 'yolov5' in model_path: 
            self.yolov5_model = Model_v2(yaml_path, nc=num_classes)                    
            #elif 'yolov4' in model_path:           
            #    self.yolov5_model = Model_u5(yaml_path, nc=num_classes)                     
            #else:
            #    raise ValueError('wrong version',model_path) 
            self.yolov5_model.load_state_dict(torch.load(model_path, map_location=self.DEVICE),strict=False)
            self.yolov5_model.float().eval()
            self.yolov5_model.to(self.DEVICE)
            self.yolov5_model.eval()
        else:
            import onnxruntime
            #onnxruntime.set_default_logger_severity(0)
            self.sess = onnxruntime.InferenceSession(model_path)
            # self.sess.set_providers(['CUDAExecutionProvider'])
            self.input_name = self.sess.get_inputs()[0].name
            self.onnx_batch_size = self.sess.get_inputs()[0].shape[0]
            self.onnx_img_size_h = self.sess.get_inputs()[0].shape[2]
            self.onnx_img_size_w = self.sess.get_inputs()[0].shape[3]         
        self.anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]] #yolov5            
        print('self.vanish_point',self.vanish_point)                                                               
        
 
    def run(self, img_path):
        """
        inputs : 
            img_path : path of the image
        outputs :
            dets : list 
        """
        self.yolov5_model.eval()
        with torch.no_grad():
            img, im0 = Yolov5_preprocess(img_path, self.DEVICE, self.imgsz_h, self.imgsz_w)
            if next(self.yolov5_model.parameters()).is_cuda:
                img = img.type(torch.cuda.FloatTensor)
            else:
                img = img.type(torch.FloatTensor)
            pred = self.yolov5_model(img, augment=False)
            img_shape, im0_shape = img.shape, im0.shape               
            dets = Yolov5_postprocess_sig(pred,img_shape, im0_shape, self.conf_thres, self.iou_thres, self.top_k_num, self.grids, self.num_classes, self.anchors,self.vanish_point)
            return dets

    def run_test(self, img_path):
        """
        inputs : 
            img_path : path of the image
        outputs :
            dets : list 
        """
        self.yolov5_model.eval()
        with torch.no_grad():
            img, im0 = Yolov5_preprocess(img_path, self.DEVICE, self.imgsz_h, self.imgsz_w)
            if next(self.yolov5_model.parameters()).is_cuda:
                img = img.type(torch.cuda.FloatTensor)
            else:
                img = img.type(torch.FloatTensor)            
            save_dir = '../applications/pytorch_batch1'
            file_name = img_path.split('/')[-1][:-4]
            save_path = open(os.path.join(save_dir, file_name+'_preprocess_img.txt'),'w')
            print(img,file=save_path)
            pred = self.yolov5_model(img, augment=False)
            save_path = open(os.path.join(save_dir, file_name+'_model_pred.txt'),'w')
            print(pred,file=save_path)
            img_shape, im0_shape = img.shape, im0.shape               
            dets = Yolov5_postprocess_sig(pred,img_shape, im0_shape, self.conf_thres, self.iou_thres, self.top_k_num, self.grids, self.num_classes, self.anchors)
            save_path = open(os.path.join(save_dir, file_name+'_postprocess_dets.txt'),'w')
            print(dets,file=save_path)             
            return dets            

    def run_onnx(self, img_path):
        """
        inputs : 
            img_path : path of the image
        outputs :
            dets : list 
        """
        with torch.no_grad():
            img, im0 = Yolov5_preprocess(img_path, self.DEVICE, self.imgsz_h, self.imgsz_w)
            np_images = np.array(img.cpu())
            np_images = np_images.astype(np.float32)
            pred_onnx = self.sess.run(None, {self.input_name: np_images })
            img_shape, im0_shape = img.shape, im0.shape 
            # print('img_shape',img_shape)
            # print('im0_shape', im0_shape)
            dets_onnx = Yolov5_postprocess_onnx_sig(pred_onnx,img_shape, im0_shape, self.conf_thres, self.iou_thres, self.top_k_num, self.grids, self.num_classes, self.anchors,self.vanish_point)
            return dets_onnx

    def run_onnx_test(self, img_path):
        """
        inputs : 
            img_path : path of the image
        outputs :
            dets : list 
        """
        with torch.no_grad():
            img, im0 = Yolov5_preprocess(img_path, self.DEVICE, self.imgsz_h, self.imgsz_w)
            save_dir = '../applications/onnx_batch1'
            file_name = img_path.split('/')[-1][:-4]
            save_path = open(os.path.join(save_dir, file_name+'_preprocess_img.txt'),'w') 
            print(img,file=save_path)           
            np_images = np.array(img.cpu())
            np_images = np_images.astype(np.float32)
            pred_onnx = self.sess.run(None, {self.input_name: np_images })
            save_path = open(os.path.join(save_dir, file_name+'_model_pred_onnx.txt'),'w')
            print(pred_onnx,file=save_path)            
            img_shape, im0_shape = img.shape, im0.shape 
            dets_onnx = Yolov5_postprocess_onnx_sig(pred_onnx,img_shape, im0_shape, self.conf_thres, self.iou_thres, self.top_k_num, self.grids, self.num_classes, self.anchors)
            save_path = open(os.path.join(save_dir, file_name+'_postprocess_dets_onnx.txt'),'w')
            print(dets_onnx,file=save_path)              
            return dets_onnx


