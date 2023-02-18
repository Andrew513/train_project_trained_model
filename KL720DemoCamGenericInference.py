# ******************************************************************************
#  Copyright (c) 2021. Kneron Inc. All rights reserved.                        *
# ******************************************************************************

import os
import sys
import platform
import argparse
import time
import threading
import sqlite3
from collections import deque


classId = []
top_left_x = []
top_left_y = []
bot_right_x = []
bot_right_y = []
#connecting database
# conn = sqlite3.connect('boundingBox.db')
# c = conn.cursor()

# c.execute("CREATE TABLE bbox (class_id int,x1 float,y1 float,x2 float,y2 float)")

# conn.commit()

# conn.close()

PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(PWD, '..'))

from utils.ExampleHelper import get_device_usb_speed_by_port_id
from utils.ExamplePostProcess import post_process_yolo_v5
import kp
import cv2

MODEL_FILE_PATH = os.path.join(PWD, '../../res/models/KL720/YoloV5s_640_640_3/models_720.nef')
_LOCK = threading.Lock()
_SEND_RUNNING = True
_RECEIVE_RUNNING = True

_image_to_inference = None
_image_to_show = None

_generic_raw_image_header = kp.GenericRawImageHeader()


#consecutive 3 photo that overlap
obstacle_detection = deque([0,0,0])
def obstacle_check():
    obstacle_detection.appendleft(1)
    obstacle_detection.pop()
    x_sum = 0;
    for x in obstacle_detection:
        x_sum += x 
    if x-sum == 3:
        print("obstacle detected")


def obs_overlap():
    for i in range (len(classId)):
        if classId[i] == 80:
            rail_right_x = bot_right_x[i]
            rail_left_x = top_left_x[i]
            rail_top_y = top_left_y[i]
            rail_bot_y = bot_right_y[i]

    # #if non-rail stuff on the way
    for i in range (len(classId)):
        if classId[i] != 80:
            item_right_x = bot_right_x[i]
            item_left_x = top_left_x[i]
            item_top_y = top_left_y[i]
            item_bot_y = bot_right_y[i]
            #scenario 1
            if item_left_x<=rail_left_x and item_right_x<=rail_right_x and item_right_x>=rail_left_x and item_bot_y >= rail_bot_y and item_bot_y <= rail_bot_y :
                obstacle_check()
            #scenario 2  
            if item_left_x>=rail_left_x and item_right_x>=rail_right_x and item_left_x<=rail_right_x and item_bot_y >= rail_bot_y and item_bot_y <= rail_bot_y :
                obstacle_check()
            #scenario 3
            if item_left_x>=rail_left_x and item_right_x<=rail_right_x and item_bot_y >= rail_bot_y and item_bot_y <= rail_bot_y :
                obstacle_check()
            #scenario 4
            if item_left_x<=rail_left_x and item_right_x>=rail_right_x and item_bot_y >= rail_bot_y and item_bot_y <= rail_bot_y :
                obstacle_check()




def _image_send_function(_device_group: kp.DeviceGroup) -> None:
    global _image_to_inference, _generic_raw_image_header, _SEND_RUNNING

    try:
        # set camera configuration
        if 'Windows' == platform.system():
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(0)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        try_cap_time = 0

        _generic_raw_image_header.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        _generic_raw_image_header.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while _SEND_RUNNING:
            try:
                # read frame from camera
                if cap.isOpened():
                    ret, _image_to_inference = cap.read()

                    if not ret and try_cap_time < 50:
                        try_cap_time += 1
                        time.sleep(0.1)
                        continue
                    elif not ret:
                        print('Error: opencv open camera failed')
                        break
                    else:
                        try_cap_time = 0
                else:
                    print('Error: opencv open camera failed')
                    break
            except:
                print('Error: opencv read frame failed')
                break

            # convert color space to RGB565
            inference_image = cv2.cvtColor(src=_image_to_inference, code=cv2.COLOR_BGR2BGR565)

            # inference image
            if ret:
                kp.inference.generic_raw_inference_send(device_group=_device_group,
                                                        image=inference_image,
                                                        image_format=kp.ImageFormat.KP_IMAGE_FORMAT_RGB565,
                                                        generic_raw_image_header=_generic_raw_image_header)

        cap.release()
    except kp.ApiKPException as exception:
        print(' - Error: inference failed, error = {}'.format(exception))

    _SEND_RUNNING = False


def _result_receive_function(_device_group: kp.DeviceGroup,
                             _model_nef_descriptor: kp.ModelNefDescriptor) -> None:
    global _image_to_show, _generic_raw_image_header, _RECEIVE_RUNNING
    _loop = 0
    _fps = 0

    while _RECEIVE_RUNNING:

        #connecting database
        conn = sqlite3.connect('boundingBox.db')
        c = conn.cursor()

        c.execute("CREATE TABLE bbox (class_id int,x1 float,y1 float,x2 float,y2 float)")


        time_start = time.time()

        # receive inference result
        try:
            generic_raw_result = kp.inference.generic_raw_inference_receive(device_group=_device_group,
                                                                            generic_raw_image_header=_generic_raw_image_header,
                                                                            model_nef_descriptor=_model_nef_descriptor)
        except kp.ApiKPException as exception:
            print(' - Error: inference failed, error = {}'.format(exception))
            exit(0)

        _loop += 1

        with _LOCK:
            temp_image = _image_to_inference.copy()

        # retrieve inference node output
        inf_node_output_list = []
        for node_idx in range(generic_raw_result.header.num_output_node):
            inference_float_node_output = kp.inference.generic_inference_retrieve_float_node(node_idx=node_idx,
                                                                                             generic_raw_result=generic_raw_result,
                                                                                             channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_CHW)
            inf_node_output_list.append(inference_float_node_output)

        # do post-process
        _yolo_result = post_process_yolo_v5(inference_float_node_output_list=inf_node_output_list,
                                            image_width=_generic_raw_image_header.width,
                                            image_height=_generic_raw_image_header.height,
                                            thresh_value=0.15,
                                            with_sigmoid=False)

        # draw bounding box
        for yolo_result in _yolo_result.box_list:
            b = 100 + (25 * yolo_result.class_num) % 156
            g = 100 + (80 + 40 * yolo_result.class_num) % 156
            r = 100 + (120 + 60 * yolo_result.class_num) % 156
            color = (b, g, r)

            cv2.rectangle(img=temp_image,
                          pt1=(int(yolo_result.x1), int(yolo_result.y1)),
                          pt2=(int(yolo_result.x2), int(yolo_result.y2)),
                          color=color,
                          thickness=3)
            #info
            class_id = yolo_result.class_num
            x1 = int(yolo_result.x1)
            y1 = int(yolo_result.y1)
            x2 = int(yolo_result.x2)
            y2 = int(yolo_result.y2)

            c.execute("INSERT INTO bbox (class_id, x1, y1, x2, y2) VALUES(?, ?, ?, ?, ?)",(class_id, x1, y1, x2, y2))

        conn.commit()

        #get all items that updated to db put into lists
        c.execute("SELECT * FROM bbox")
        items = c.fetchall()

        for item in items:
            classId.append(item[0])
            top_left_x.append(item[1])
            top_left_y.append(item[2])
            bot_right_x.append(item[3])
            bot_right_y.append(item[4])

        obs_overlap()
        c.execute("DROP TABLE bbox")
        conn.commit()
        c.close()
        conn.close()

        time_end = time.time()

        if 30 == _loop:
            _fps = 1 / (time_end - time_start)
            _loop = 0

        # draw FPS
        cv2.putText(img=temp_image,
                    text='FPS: {:.2f}'.format(_fps),
                    org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1,
                    color=(200, 200, 200),
                    thickness=1,
                    lineType=cv2.LINE_AA)
        cv2.putText(img=temp_image,
                    text='Press \'ESC\' to exit',
                    org=(10, temp_image.shape[0] - 10),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1,
                    color=(200, 200, 200),
                    thickness=1,
                    lineType=cv2.LINE_AA)

        with _LOCK:
            _image_to_show = temp_image.copy()

    _RECEIVE_RUNNING = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KL520 Demo Camera Generic Inference Example.')
    parser.add_argument('-p',
                        '--port_id',
                        help='Using specified port ID for connecting device (Default: port ID of first scanned Kneron '
                             'device)',
                        default=0,
                        type=int)
    args = parser.parse_args()

    usb_port_id = args.port_id

    """
    check device USB speed (Recommend run KL720 at super speed)
    """
    try:
        if kp.UsbSpeed.KP_USB_SPEED_SUPER != get_device_usb_speed_by_port_id(usb_port_id=usb_port_id):
            print('\033[91m' + '[Error] Device is not run at super speed.' + '\033[0m')
            exit(0)
    except Exception as exception:
        print('Error: check device USB speed fail, port ID = \'{}\', error msg: [{}]'.format(usb_port_id,
                                                                                             str(exception)))
        exit(0)

    """
    connect the device
    """
    try:
        print('[Connect Device]')
        device_group = kp.core.connect_devices(usb_port_ids=[usb_port_id])
        print(' - Success')
    except kp.ApiKPException as exception:
        print('Error: connect device fail, port ID = \'{}\', error msg: [{}]'.format(usb_port_id,
                                                                                     str(exception)))
        exit(0)

    """
    setting timeout of the usb communication with the device
    """
    print('[Set Device Timeout]')
    kp.core.set_timeout(device_group=device_group, milliseconds=10000)
    print(' - Success')

    """
    upload model to device
    """
    try:
        print('[Upload Model]')
        model_nef_descriptor = kp.core.load_model_from_file(device_group=device_group,
                                                            file_path=MODEL_FILE_PATH)
        print(' - Success')

        print('[Model NEF Information]')
        print(model_nef_descriptor)
    except kp.ApiKPException as exception:
        print('Error: upload model failed, error = \'{}\''.format(str(exception)))
        exit(0)

    """
    prepare app generic inference config
    """
    _generic_raw_image_header = kp.GenericRawImageHeader(
        model_id=model_nef_descriptor.models[0].id,
        resize_mode=kp.ResizeMode.KP_RESIZE_ENABLE,
        padding_mode=kp.PaddingMode.KP_PADDING_CORNER,
        normalize_mode=kp.NormalizeMode.KP_NORMALIZE_KNERON
    )

    """
    starting inference work
    """
    print('[Starting Inference Work]')
    print(' - Starting inference')
    send_thread = threading.Thread(target=_image_send_function, args=(device_group,))
    receive_thread = threading.Thread(target=_result_receive_function, args=(device_group, model_nef_descriptor))

    send_thread.start()
    receive_thread.start()

    cv2.namedWindow('Generic Inference', cv2.WND_PROP_ASPECT_RATIO or cv2.WINDOW_GUI_EXPANDED)
    cv2.setWindowProperty('Generic Inference', cv2.WND_PROP_ASPECT_RATIO, cv2.WND_PROP_ASPECT_RATIO)

    while True:
        with _LOCK:
            if None is not _image_to_show:
                cv2.imshow('Generic Inference', _image_to_show)

        if (27 == cv2.waitKey(10)) or (not _SEND_RUNNING) or (not _RECEIVE_RUNNING):
            break

    cv2.destroyAllWindows()

    _SEND_RUNNING = False
    _RECEIVE_RUNNING = False

    send_thread.join()
    receive_thread.join()
