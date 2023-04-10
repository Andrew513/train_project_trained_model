# coding: utf-8
import torch
import torchvision
import time
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, top_k_num=3000, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    # print('conf_thres',conf_thres)
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero().t()
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]



        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        # # Sort by confidence
        ind_Sort_by_confidence = x[:, 4].argsort(descending=True)
        boxes = boxes[ind_Sort_by_confidence][:top_k_num] #
        scores = scores[ind_Sort_by_confidence][:top_k_num] #
        x = x[ind_Sort_by_confidence][:top_k_num] #
        # cross classes nms
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        # if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
        #     update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
        #     iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
        #     weights = iou * scores[None]  # box weights
        #     x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
        #     if redundant:
        #         i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output

def non_max_suppression_kneron(prediction, conf_thres=0.1, iou_thres=0.6, top_k_num=3000, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero().t()
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]



        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        # Sort by confidence
        ind_Sort_by_confidence = x[:, 4].argsort(descending=True)
        boxes = boxes[ind_Sort_by_confidence][:top_k_num] #
        scores = scores[ind_Sort_by_confidence][:top_k_num] #
        x = x[ind_Sort_by_confidence][:top_k_num] #
        # cross classes nms
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]


        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

def scale_coords_ori(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        #pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        #pad = ratio_pad[1]

    # coords[:, [0, 2]] -= pad[0]  # x padding
    # coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def scale_coords_test(img1_shape, coords, img0_shape, ratio_pad=None):

    coords[:, 0] /= (img1_shape[1] / img0_shape[1])
    coords[:, 2] /= (img1_shape[1] / img0_shape[1])
    coords[:, 1] /= (img1_shape[0] / img0_shape[0])
    coords[:, 3] /= (img1_shape[0] / img0_shape[0])
    clip_coords(coords, img0_shape)
    return coords

def Yolov5_postprocess(pred, img_shape, im0_shape, conf_thres, iou_thres, top_k_num, num_classes, vanish_point) :
    classes, agnostic_nms = None, False#
    img_h = im0_shape[0]
    vanish_y2 = vanish_point * float(img_h)
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, top_k_num, classes=classes, agnostic=agnostic_nms)
    #return pred
    dets = []
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(im0_shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img_shape[2:], det[:, :4], im0_shape).round()
            det = det[det[:,3]>=vanish_y2]
            # (x1,y1,x2,y2) -> (x1,y1,w,h) for public_field.py
            det[:, 2] = det[:, 2] - det[:, 0] 
            det[:, 3] = det[:, 3] - det[:, 1]  
            det = det.cpu().numpy()  
            dets.append(det)                                
    
    if dets and len(dets) > 0:
        dets = np.asarray(dets)
        dets = np.squeeze(dets, axis=0) # remove outer []
        dets = dets.tolist()

    return dets

def make_grid(nx=20, ny=20):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    grids = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
    return grids

def Yolov5_postprocess_onnx_sig(out,img_shape, im0_shape, conf_thres, iou_thres, top_k_num, grids, num_classes, anchors,vanish_point) :
    nc = num_classes  # number of classes
    no = nc + 5  # number of outputs per anchor    
    nl = len(anchors)  # number of detection layers
    na = len(anchors[0]) // 2  # number of anchors
    a = torch.tensor(anchors).float().view(3, -1, 2)
    anchor_grid = a.clone().view(3, 1, -1, 1, 1, 2)
    stride = torch.tensor([ 8., 16., 32.])     
    z = []
    for i in range(nl):
        x = torch.from_numpy(out[i])
        # print('x.shape',x.shape)
        bs, _, ny, nx = x.shape  # x(bs,3,20,20,85)
        x = x.view(bs, na, no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        # grid_r = make_grid(nx, ny) ##grid
        # grid_r = grid_r.numpy() ##grid
        # file_name = str(i)+'.npy' ##grid
        # np.save(file_name,grid_r) ##grid
        grid = grids[i]#
        #y = x.sigmoid()
        y = x
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid) * stride[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
        z.append(y.view(bs, -1, no))

    pred = torch.cat(z, 1)
    return Yolov5_postprocess(pred, img_shape, im0_shape, conf_thres, iou_thres, top_k_num, num_classes,vanish_point)

def Yolov5_postprocess_sig(out,img_shape, im0_shape, conf_thres, iou_thres, top_k_num, grids, num_classes, anchors,vanish_point) :  
    nc = num_classes  # number of classes
    no = nc + 5  # number of outputs per anchor    
    nl = len(anchors)  # number of detection layers
    na = len(anchors[0]) // 2  # number of anchors
    a = torch.tensor(anchors).float().view(3, -1, 2)
    anchor_grid = a.clone().view(3, 1, -1, 1, 1, 2).to(out[0].device)
    stride = torch.tensor([ 8., 16., 32.]).to(out[0].device)     
    z = []
    for i in range(nl):
        x = out[i]
        bs, _, ny, nx = x.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        # print('x.shape',x.shape)
        x = x.view(bs, na, no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        
        # grid_r = make_grid(nx, ny) ##grid
        # grid_r = grid_r.numpy() ##grid
        # file_name = str(i)+'.npy' ##grid
        # np.save(file_name,grid_r) ##grid
        
        grid = grids[i].to(out[0].device) #
        #y = x.sigmoid()
        y = x
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid) * stride[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
        z.append(y.view(bs, -1, no))   
    # exit(0)
    pred = torch.cat(z, 1)
    return Yolov5_postprocess(pred, img_shape, im0_shape, conf_thres, iou_thres, top_k_num, num_classes,vanish_point)