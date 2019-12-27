import os
from threading import Thread
import struct  # to send `int` as  `4 bytes`
import time   # for test
import cv2
from imutils.video import VideoStream
import datetime
import argparse
import base64
import timeit
import shutil
from utils_side_view import processing_view, load_model_ctnr_no, processing_cnt_number, encode_image64
import json
from check_digit_container import check_container_id, check_container_iso, split_id_iso, get_closest_iso
import requests
import socketio

# sio = socketio.Client()
# sio.connect('http://localhost:8000')

list_ids_detected_global = []


def push_result(result, socket_result):
    list_result = result
    jsonPayload = {"data": list_result}
    jsonPayload = json.dumps(jsonPayload)
    headers = {'Content-Type': 'application/json'}
    r = requests.post(socket_result, data=jsonPayload, headers=headers)
    return r


def recreate_folder(folder):
    if os.path.isdir(folder) == True:
        shutil.rmtree(folder)
    os.makedirs(folder)


def save_folder_time():
    now = datetime.datetime.now()
    save_time = str(now.year) + "_" + str(now.month) + "_" + str(now.day) + \
        "_" + str(now.hour) + "_" + str(now.minute) + "_" + str(now.second)
    # spvs
    folder_save = '/media/newhd/new_data/SPVSResult/QC04/cont/' + save_time + "/"
    # folder_save = '/data/rdteam/cont/' + save_time + "/"
    recreate_folder(folder_save)
    return folder_save


def show_cam(frame, cam_name):
    frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
    result = {
        "cam": cam_name,
        "image": encode_image64(frame)}

    jsonPayload = json.dumps(result)
    headers = {'Content-Type': 'application/json'}
    r = requests.post("http://localhost:5000/api/home/stream-camera",
                      data=jsonPayload, headers=headers)


# 2.0TB->SPVSResult->QC04->20191203-
def get_image_cont(result, frame1, net, meta, cam_name, folder_save):

    global list_ids_detected_global
    for index, item in enumerate(result):
        if item["Class"] == "ContainerNo":
            loc_no = item["Position"]
            container_no_img = frame1[loc_no[1] - loc_no[3]//2 - loc_no[3]//20: loc_no[1] +
                                      loc_no[3]//2 + loc_no[3]//20, loc_no[0] - loc_no[2]//2 - loc_no[2]//20: loc_no[0] + loc_no[2]//2 + loc_no[2]//20]

            result_ocr = processing_cnt_number(container_no_img, net, meta)
            cont_id, cont_iso = split_id_iso(result_ocr[0])

            if check_container_id(cont_id) == True:
                if cont_id not in list_ids_detected_global:
                    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    img_cont_save = folder_save + \
                        current_time + ".jpg"
                    # save image result
                    cv2.imwrite(img_cont_save, container_no_img)

                    # correct iso
                    cont_iso = get_closest_iso(cont_iso)

                    result_cont = [{
                        "image": img_cont_save,
                        "cntr_iso": cont_iso,
                        "status": "LD",
                        "time": current_time,
                        "cntr_id": cont_id,
                        "cam_num": cam_name,
                        "lane": "5"
                    }]

                    print(result_cont)
                    push_result(
                        result_cont, " http://localhost:5000/api/container/result")
                    list_ids_detected_global.append(cont_id.strip())

                    # biz logic
                    # container back or front
                    cam_name = str(cam_name)
                    container_view = "front"
                    if cam_name == "5" or cam_name == "7":
                        container_view = "back"
                    if cam_name == "6" or cam_name == "8":
                        container_view = "front"

                    json_meta = {"container_ids_on_spreader":
                                 {"container_id": cont_id, "container_iso": cont_iso, "container_image": img_cont_save, "containerview": container_view}}
                    # sio.emit('side_cameras_info', json_meta)


def only_show_cam(video_capture1, resize_camera, cam_name):
    while True:
        try:
            # spvs
            frame1 = video_capture1.read()
            # ret1, frame1 = video_capture1.read()
            frame1 = cv2.resize(
                frame1, None, fx=resize_camera, fy=resize_camera)
            show_cam(frame1, cam_name)
            time.sleep(0.5)
        except:
            print("exception")


def parse_args():
    parser = argparse.ArgumentParser(description='Recognition System')
    # parameter
    # rtsp link rtsp://admin:dou123789@10.0.0.11:554/cam/realmonitor?channel=1&subtype=0
    parser.add_argument(
        '--rtsp', default='rtsp://admin:SPVS@@15411@192.168.2.28/profile3/media.smp', help='rtsp link')
    parser.add_argument('--crop-camera', default='0,1,0.5,1',
                        help='crop camera height and width')
    parser.add_argument('--resize-camera', default='0.35',
                        help='resize camera height and width')
    parser.add_argument('--cam-name', default='8', help='cam name')
    parser.add_argument('--only-show', default='0',
                        help='only show cam or recognition')

    args = parser.parse_args()
    return args


class Streaming(Thread):
    def __init__(self, args):
        Thread.__init__(self)
        self.args = args

        self.cam_name = args.cam_name
        self.resize_camera = float(args.resize_camera)
        self.crop_camera = args.crop_camera.split(',')
        self.only_show = args.only_show

        # spvs
        # self.video_capture1 = cv2.VideoCapture(args.rtsp)
        self.video_capture1 = VideoStream(src=args.rtsp).start()

    # remember try catch
    # remember get logic id iso push
    # remember show cam

    def process_camera(self):

        if self.only_show == "1":
            only_show_cam(self.video_capture1,
                          self.resize_camera, self.cam_name)

        else:
            net_ctnr, meta_ctnr = load_model_ctnr_no()
            folder_save = save_folder_time()

            count = 0
            while True:
                try:
                    # spvs
                    frame1 = self.video_capture1.read()
                    # ret1, frame1 = self.video_capture1.read()

                    frame1 = cv2.resize(
                        frame1, None, fx=self.resize_camera, fy=self.resize_camera)
                    if count % 100 == 0:
                        show_cam(frame1, self.cam_name)
                        count = 0

                    h, w, c = frame1.shape
                    frame1 = frame1[int(float(self.crop_camera[0]*h)):int(float(self.crop_camera[1])*h),
                                    int(float(self.crop_camera[2])*w):int(float(self.crop_camera[3])*w)]

                    result = processing_view(
                        frame1, net_ctnr, meta_ctnr, None, None, index=1)
                    get_image_cont(result, frame1, net_ctnr,
                                   meta_ctnr, self.cam_name, folder_save)

                    count += 1
                except:
                    print("exception")


if __name__ == '__main__':
    # args = None
    args = parse_args()
    Streaming(args).process_camera()

    # python3 recognition_cam.py --rtsp 'rtsp://admin:SPVS@@15411@192.168.2.28/profile3/media.smp' --crop-camera '0,1,0.5,1' --resize-camera 0.35 --cam-name 8
    # python3 recognition_cam.py --rtsp '/data/YRCVideo/1202 lane 5/2019_12_2_4_6_48(Lane5)/cam6.avi' --crop-camera '0,1,0.5,1' --resize-camera 0.35 --cam-name 6 --only-show 0
