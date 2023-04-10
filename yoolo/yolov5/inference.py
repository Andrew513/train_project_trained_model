import os
import sys
import argparse
import yaml
import cv2
import numpy as np

def draw(img_path, bboxes, save_path = None, names = None):
    
    img = cv2.imread(img_path)
    for bbox in bboxes:
        l,t,w,h,score,class_id=bbox
        if names is not None:
            class_id = names[int(class_id)]
        img = cv2.rectangle(img,(int(l),int(t)),(int(l+w),int(t+h)),(0, 255, 0),6)
        text = "{}".format(class_id) + "  {}".format(np.round(score, 3))
        img = cv2.putText(img, text, (int(l), int(t)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    if save_path is None:
        save_path = img_path
    cv2.imwrite(save_path, img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str, default=None, help='path to image')
    parser.add_argument('--save-path', type=str, default=None, help='path to save image')
    parser.add_argument('--data', type=str, default='data/pretrained_paths_520.yaml', help='the path to pretrained model paths yaml file')
    parser.add_argument('--conf_thres', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.5, help='iou threshold for NMS')
    parser.add_argument('--onnx', help='inference onnx model',action='store_true')
    
    args = parser.parse_args()
    
    par_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    sys.path.append(par_path)
    sys.path.append(os.path.join(par_path, 'exporting') )

    from yolov5.yolov5_runner import Yolov5Runner

    with open(args.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
        
    num_classes = data_dict['nc']
    input_w = data_dict['input_w']
    input_h = data_dict['input_h']
    grid_dir = data_dict['grid_dir']
    grid20_path = data_dict['grid20_path']
    grid40_path = data_dict['grid40_path']
    grid80_path = data_dict['grid80_path']
    path = data_dict['path']
    
    yolov5_model = Yolov5Runner(model_path=data_dict['pt_path'], yaml_path=data_dict['yaml_path'], grid20_path=grid20_path, grid40_path=grid40_path, grid80_path=grid80_path, num_classes=num_classes, imgsz_h=input_h, imgsz_w=input_w, conf_thres=args.conf_thres, iou_thres=args.iou_thres, top_k_num=3000, vanish_point=0.0) 
    if args.onnx:
        bboxes = yolov5_model.run_onnx(args.img_path)
    else:
        bboxes = yolov5_model.run(args.img_path)
    
    print(bboxes)
    
    if args.save_path is not None:
        draw(args.img_path, bboxes, save_path = args.save_path, names = data_dict['names'])
