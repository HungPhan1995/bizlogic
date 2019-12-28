import os
from threading import Thread
import time   # for test
import cv2
from imutils.video import VideoStream
import datetime
import base64
import json
import requests
import socketio

sio_lane = socketio.Client()
sio_lane.connect('http://localhost:5000')

root_folder_save = '/media/newhd/new_data/SPVSResult/QC04/'


current_working_lane = 5

@sio_lane.on('laneChange', namespace='')
def detect_lane(data):
    global current_working_lane
    current_working_lane = int(data['lane'])


def encode_image64(image):
    retval, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer)
    return "data:image/jpg;base64," + str(jpg_as_text.decode('utf-8'))

def show_cam(frame, cam_name):
	frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
	result = {
		"cam": cam_name,
		"image": encode_image64(frame)}

	jsonPayload = json.dumps(result)
	headers = {'Content-Type': 'application/json'}
	try:
		r = requests.post("http://localhost:5000/api/home/stream-camera",
						  data=jsonPayload, headers=headers)
	except:
		print('Exception Huy port 5000')

class Streaming(Thread):
	def __init__(self, args):
		Thread.__init__(self)
		self.args = args
		self.cam_name_front = 3
		self.cam_name_back = 1
		self.video_capture_cam1 = VideoStream(src='rtsp://admin:SPVS@@15411@192.168.2.21/profile3/media.smp').start()
		self.video_capture_cam2 = VideoStream(src='rtsp://admin:SPVS@@15411@192.168.2.22/profile3/media.smp').start()
		self.video_capture_cam3 = VideoStream(src='rtsp://admin:SPVS@@15411@192.168.2.23/profile3/media.smp').start()
		self.video_capture_cam4 = VideoStream(src='rtsp://admin:SPVS@@15411@192.168.2.24/profile3/media.smp').start()
	def process_streaming(self):
		current_second = -1
		old_second = -1
		while True: 
			try:
				if current_working_lane in [4,5,6]:
					self.cam_name_front = 3
					self.cam_name_back = 1
					image_front = self.video_capture_cam3.read()
					image_back = self.video_capture_cam1.read()
				elif current_working_lane in [1,2,3]:
					self.cam_name_front = 4
					self.cam_name_back = 2
					image_front = self.video_capture_cam4.read()
					image_back = self.video_capture_cam2.read()
				now = datetime.datetime.now()
				current_second = now.second
				if current_second % 5 == 0 and old_second != current_second:
					old_second = current_second
					show_cam(image_front,self.cam_name_front)
					show_cam(image_back,self.cam_name_back)
			except:
				print("There are something wrong show other camera")
if __name__ == '__main__':
	args = None
	Streaming(args).process_streaming()