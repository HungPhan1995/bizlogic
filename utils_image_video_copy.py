from ctypes import *
import math
from random import randint
import cv2
import os
import shutil
import copy
import datetime
import sys
import hashlib
import numpy as np
import json
import operator
import re
import pandas as pd
import requests
# from check_digit_container import check_container_id
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from tensorflow.keras.preprocessing.image import ImageDataGenerator
gen = ImageDataGenerator()
import base64
import threading
import datetime

import keras

from PIL import Image

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]



#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int,
                              c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def array_to_image(arr):
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2, 0, 1)
    c, h, w = arr.shape[0:3]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w, h, c, data)
    return im, arr


def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    if isinstance(image, bytes):
        im = load_image(image, 0, 0)
    else:
        im, image = array_to_image(image)
        rgbgr_image(im)

    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh,
                             hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms):
        do_nms_obj(dets, num, meta.classes, nms)

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append(
                    (meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    if isinstance(image, bytes):
        free_image(im)
    free_detections(dets, num)
    return res


def load_model():
    # loading model cnt
    net = load_net(b"weights_copy/cntr_full.cfg",
                   b"weights_copy/cntr_full.weights", 0)
    meta = load_meta(b"weights_copy/cntr_full.data")

    return net, meta

def load_model_ctnr_no():
    # net = load_net(b"darknet/cfg_cntr_full/yolov3-tiny.cfg",
    #                 b"darknet/cfg_cntr_full/yolov3-tiny_last-181019.weights", 0)
    net = load_net(b"weights_copy/yolov3-tiny.cfg",
                    b"weights_copy/yolov3-tiny_last-40classes.weights", 0)
    meta = load_meta(b"weights_copy/trainer.data")

    return net, meta

def load_model_truck():
    # net = load_net(b"darknet/cfg_cntr_full/yolov3-tiny.cfg",
    #                 b"darknet/cfg_cntr_full/yolov3-tiny_last-181019.weights", 0)
    net = load_net(b"weights_copy/yolov3-tiny-truck.cfg",
                    b"weights_copy/yolov3-tiny-truck_last.weights", 0)
    meta = load_meta(b"weights_copy/trainer_truck.data")

    return net, meta

def load_model_door():
    model = keras.models.load_model('weights_copy/containers-25-0.97.h5')
    return model


def checkOutside(target, obj):
    targetLeft = target['x']
    targetTop = target['y']
    targetWidth = target['w']
    targetHeight = target['h']

    objLeft = obj['x']
    objTop = obj['y']
    objWidth = obj['w']
    objHeight = obj['h']

    if targetLeft > (objLeft + objWidth) or (targetLeft + targetWidth) < objLeft or targetTop > (objTop + objHeight) or (targetTop + targetHeight) < objTop:
        return True
    else:
        d1x = targetLeft
        d1y = targetTop
        d1xMax = targetLeft + targetWidth
        d1yMax = targetTop + targetHeight
        d2x = objLeft
        d2y = objTop
        d2xMax = objLeft + objWidth
        d2yMax = objTop + objHeight

        x_overlap = max(0, min(d1xMax, d2xMax) - max(d1x, d2x))
        y_overlap = max(0, min(d1yMax, d2yMax) - max(d1y, d2y))

        area = x_overlap * y_overlap
        if (area / (targetWidth * targetHeight)) < 0.5:
            return True
        else:
            return False


def convert(raw_arr, frame):
    rslt = []
    for item in raw_arr:
        rslt.append([
            item[0].decode("utf-8"),
            item[1],
            [
                item[2][0],
                item[2][1],
                item[2][2],
                item[2][3]
            ]
        ])

    height = frame.shape[0]
    width = frame.shape[1]

    for item in rslt:
        if (item[2][0] + item[2][2] / 2) > width:
            item[2][2] = (width - item[2][0]) * 2
        if item[2][0] - item[2][2] / 2 < 0:
            item[2][0] = item[2][2] / 2
        if (item[2][1] + item[2][3] / 2) > height:
            item[2][3] = (height - item[2][1]) * 2
        if item[2][1] - item[2][3] / 2 < 0:
            item[2][1] = item[2][3] / 2

    rslt2 = []
    for item in rslt:
        c = item[2]
        # x = int(c[0] - c[2] / 2)
        # y = int(c[1] - c[3] / 2)
        x = int(c[0])
        y = int(c[1])
        w = int(c[2])
        h = int(c[3])
        className = item[0]
        rslt2.append([
            className, x, y, w, h, item[1]
        ])
    return rslt2

def drawCandidate(arr, frame):
    font = cv2.FONT_HERSHEY_PLAIN

    for item_cnt in arr:
        className = item_cnt[0]
        x = item_cnt[1]
        y = item_cnt[2]
        w = item_cnt[3]
        h = item_cnt[4]
        conf = int(item_cnt[5] * 100)
        cnt_no_list = item_cnt[6]
        # draw container
        cv2.rectangle(frame, (x-w//2, y-h//2), (x + w//2, y + h//2), (0, 0, 255), 2)
        cv2.putText(frame, str(className) + " : " + str(conf), (x-w//2, y-h//2), font,
                    1.5, (0, 0, 255), 2, cv2.LINE_AA)

        # draw list of container number
        for item_cnt_num in cnt_no_list:
            className = item_cnt_num[0]
            x = item_cnt_num[1]
            y = item_cnt_num[2]
            w = item_cnt_num[3]
            h = item_cnt_num[4]
            conf_cnt = int(item_cnt_num[5] * 100)
            cnt_num_value = item_cnt_num[6]
            conf_num_value = int(item_cnt_num[7] * 100)
            cv2.rectangle(frame,  (x-w//2, y-h//2), (x + w//2, y + h//2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x-w//2, y-h//2 - 30), (x-w//2 + 350, y-h//2), (0, 0, 0), -1)
            cv2.putText(frame, str(cnt_num_value) + " | " + str(conf_num_value) + "%",
                        (x-w//2, y-h//2), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

def drawCandidate_top_view(arr, frame):
    font = cv2.FONT_HERSHEY_PLAIN
    for obj in arr:
        className = obj["Object"]
        x = (obj["Position"])[0]
        y = (obj["Position"])[1]
        w = (obj["Position"])[2]
        h = (obj["Position"])[3]
        cf = int(obj["Confidence"]*1000)/10
        if obj["Object"] == "TruckHead":
            truck_id , id_cf = obj["Truck ID"]
            id_cf = int(id_cf*1000)/10

        cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 0, 255), 2)
        obj_str = str(className) + " | " + str(cf) + "%" if obj["Object"] != "TruckHead" else str(className) + " - " + str(truck_id) + " | " + str(id_cf)
        cv2.putText(frame, obj_str, (x - w//2, y - h//2), font, 2.0, (0, 255, 255), 2, cv2.LINE_AA)
    return frame

def processing(frame, net, meta, net_ctnr, meta_ctnr, index):
    arr = detect(net, meta, frame)
    processing_arr = convert(arr, frame)

    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.5
    thickness = 2
    cnt_rslt = ''
    str_out = []
    # detect number
    for itemP in processing_arr:
        className = itemP[0]
        x = itemP[1]
        y = itemP[2]
        w = itemP[3]
        h = itemP[4]
        conf = itemP[5]

        if className != "ContainerNo":
            continue

        cnt_img = frame[(y-(h//2))-20:(y+(h//2))+20, (x-(w//2))-20:(x+(w//2))+20]
        cnt_rslt, conf_id = processing_cnt_number(cnt_img, net_ctnr, meta_ctnr)
        # check_container = False if not check_container_id(cnt_rslt) else True
        # cont_id = cnt_rslt + "_" + str(check_container)

        # cv2.imwrite("outPut_for_Bk/" + str(index) + "_" + cont_id + ".jpg", cnt_img)

        # size, offset = cv2.getTextSize(cont_id, font, font_scale, thickness)
        # y_offset = -30
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
        # cv2.rectangle(frame, (x, y-offset+y_offset),
        #             (x+size[0]+offset, y+size[1]+offset+y_offset), (0,0,0), cv2.FILLED)
        # cv2.putText(frame, cont_id, (x+offset, y+15+y_offset),
        #             font, font_scale, (255,255,255), thickness, cv2.LINE_AA)

        str_out.append(['Container', x, y, w, h, conf,
                        [['ContainerNo', x, y, w, h, conf, cnt_rslt, conf_id]], ''])
    return frame, str_out

def processing_top_view(frame, net, meta, save_folder = False, should_invert = False):
    
    arr = detect(net, meta, frame)
    processing_arr = convert(arr, frame)
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.5
    thickness = 2

    cnt_rslt = ''
    str_out = []

    # model_door = load_model_door()
    # detect number
    for itemP in processing_arr:
        className = itemP[0]
        x = itemP[1]
        y = itemP[2]
        w = itemP[3]
        h = itemP[4]
        conf = itemP[5]

        if className != "TruckHead":
            str_out.append({"Object": className, "Position": [x, y, w, h], "Confidence": conf})
        else:
            cnt_img = frame[(y-(h//2))-20:(y+(h//2))+20, (x-(w//2))-20:(x+(w//2))+20]
            if cnt_img.shape[0] > 0 and cnt_img.shape[1] > 0:
                truck_id, cf = processing_truck_number(cnt_img, net, meta, should_invert)
                if save_folder != False:
                    crop_img = Image.fromarray(cnt_img)
                    crop_img = crop_img.save(save_folder + truck_id + ".jpg")
                str_out.append({"Object": className, "Position": [x, y, w, h], "Confidence": conf, "Truck ID": (truck_id, cf)})
    return str_out

# def processing_door(frame, net, meta, model_door, save_folder):
#     arr = detect(net, meta, frame)
#     processing_arr = convert(arr, frame)
#     objs = []
#     for item in processing_arr:
#         className = item[0]
#         x = item[1]
#         y = item[2]
#         w = item[3]
#         h = item[4]
#         conf = item[5]

#         cnt_img_origin = np.zeros((0, 0, 0))
#         if className == "Container":
#             try:
#                 cnt_img_origin = frame[(y-(h//2))-20:(y+(h//2))+20, (x-(w//2))-20:(x+(w//2))+20]
#             except:
#                 print('Exception expand door')
#                 cnt_img_origin = frame[(y-(h//2)):(y+(h//2)), (x-(w//2)):(x+(w//2))]

#             if cnt_img_origin.shape[0] > 0:
#                 cnt_img = cv2.resize(cnt_img_origin, (300, 300), interpolation = cv2.INTER_AREA)
#                 cnt_img = cnt_img.reshape(1,300,300,3)
#                 res = model_door.predict(cnt_img)[0][0]
#                 objs.append((int(res), cnt_img_origin))
#     return objs


def processing_door_v2(frame, position, model_door):
    objs = []
    x,y,w,h = position
    cnt_img_origin = np.zeros((0, 0, 0))
    try:
        cnt_img_origin = frame[(y-(h//2))-20:(y+(h//2))+20, (x-(w//2))-20:(x+(w//2))+20]
    except:
        print('Exception expand door')
        cnt_img_origin = frame[(y-(h//2)):(y+(h//2)), (x-(w//2)):(x+(w//2))]

    if cnt_img_origin.shape[0] > 0:
        cnt_img = cv2.resize(cnt_img_origin, (300, 300), interpolation = cv2.INTER_AREA)
        cnt_img = cnt_img.reshape(1,300,300,3)
        res = model_door.predict(cnt_img)[0][0]
        objs.append((int(res), cnt_img_origin))
    return objs



def processing_view(frame, net, meta, net_ctnr, meta_ctnr, index):
    arr = detect(net, meta, frame)
    processing_arr = convert(arr, frame)

    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.5
    thickness = 2

    obj_lst = []
    # detect number
    for itemP in processing_arr:
        className = itemP[0]
        x = itemP[1]
        y = itemP[2]
        w = itemP[3]
        h = itemP[4]
        conf = itemP[5]
        obj_lst.append({"Class":className, "Position": [x, y, w, h]})
    return obj_lst

def sort_by_distant(chars, x0 = 0, y0 = 0):
    chars.sort(key=lambda char: (pow(char[1] - x0, 2) + pow(char[2] - y0, 2)));
    return chars

def sort_by_tan(chars, corner):
    chars.sort(key=lambda char: (np.arctan(((char[2] - corner[2]) if (char[2] - corner[2]) else 1)/((char[1] - corner[1]) if (char[1] - corner[1]) else 1))))
    
    tans = [(np.arctan(((char[2] - corner[2]) if (char[2] - corner[2]) else 1)/((char[1] - corner[1]) if (char[1] - corner[1]) else 1))) for char in chars] 
    
    check_tans_x = [tan for tan in tans if abs(tan) < 0.09]
    
    check_tans_y = [tan for tan in tans if 1.57 - abs(tan) < 0.09]

    if len(check_tans_x) >= 3 or len(check_tans_y) >=3:
        chars.sort(key=lambda char: abs(np.arctan(((char[2] - corner[2]) if (char[2] - corner[2]) else 1)/((char[1] - corner[1]) if (char[1] - corner[1]) else 1))))
        tans = [abs(np.arctan(((char[2] - corner[2]) if (char[2] - corner[2]) else 1)/((char[1] - corner[1]) if (char[1] - corner[1]) else 1))) for char in chars]

    equal_count = 1
    tan = tans[0]
    rows = [[chars[0]]]
    row_count = 0
    for index in range(1, len(tans)):
        if tans[index] - tan < 0.12:
            equal_count += 1
            rows[row_count].append(chars[index])

        elif equal_count < 3:
            tan = tans[index]
            equal_count = 1
            rows[row_count] = [chars[index]]

        elif equal_count >= 3:
            tan = tans[index]
            row_count += 1
            equal_count = 1
            rows.append([chars[index]])

    if(len(rows) > 1):
        rows = [row for row in rows if len(row) >= 3]
        rows = [sort_by_distant(row, corner[1], corner[2]) for row in rows]
        rows.sort(key=lambda row: pow(row[0][1]-corner[1], 2) + pow(row[0][2]-corner[2], 2))
        rows.reverse()

    row = rows[-1]

    rest = [char for char in chars if char not in row]
    return row, rest

def processing_truck_number(frame, net, meta, invert = False):
    if invert == True:
        frame = gen.apply_transform(frame, {'theta': 180})
    h, w, _ = frame.shape
    truck_rslt = detect(net, meta, frame)
    processing_truck_rslt = convert(truck_rslt, frame)
    cf = 1
    for item in processing_truck_rslt:
        cf *= item[5]
    
    if len(processing_truck_rslt) > 0:
        chars = processing_truck_rslt
        chars = sort_by_distant(chars)

        flatString = ""
        for char in chars:
            flatString += str(char[0])

        confidence = 1
        for char in processing_truck_rslt:
            confidence *= char[5]

        return flatString, confidence
    else:
        return "", 1