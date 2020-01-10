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

from utils_image_video_copy import processing, processing_view, load_model, load_model_ctnr_no, drawCandidate, drawCandidate_top_view, load_model_truck, processing_top_view, processing_door_v2, load_model_door

import json
import requests
import sys
from time import gmtime, strftime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import socketio

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sio = socketio.Client()
sio.connect('http://localhost:8000')

sio_lane = socketio.Client()
sio_lane.connect('http://localhost:5000')

root_folder_save = '/media/newhd/new_data/SPVSResult/QC04/'
# root_folder_save = '/data/Hung/bizlogic/data_save_to_test'


gen = ImageDataGenerator()

states, current_working_lane = None, 5
door_statistic = []
old_value_door = None
truck_tail_statistic = []
old_truck_tail_value = None

@sio_lane.on('laneChange', namespace='')
def detect_lane(data):
    global current_working_lane
    current_working_lane = int(data['lane'])

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

	# edit to truck or door - by your module
	folder_save = '/media/newhd/new_data/SPVSResult/QC04/truck/' + save_time + "/"
	recreate_folder(folder_save)
	return folder_save

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


# 2.0TB->SPVSResult->QC04->20191203-
def get_image_cont(result, frame1, net, meta, cam_name, folder_save):

	# process recognition here
	time.sleep(0.01)
	return


def parse_args():
	parser = argparse.ArgumentParser(description='Recognition System')
	# parameter
	# rtsp link rtsp://admin:dou123789@10.0.0.11:554/cam/realmonitor?channel=1&subtype=0
	parser.add_argument(
		'--link-video', default='', help='link video')
	# parser.add_argument('--crop-camera', default='0,1,0.5,1',
						# help='crop camera height and width')
	# parser.add_argument('--resize-camera', default='0.35',
						# help='resize camera height and width')
	# parser.add_argument('--cam-name', default='8', help='cam name')

	# parser.add_argument('--working-lane', default='4', help='working lane')

	args = parser.parse_args()
	return args


class Streaming(Thread):
	def __init__(self, args):
		Thread.__init__(self)
		self.args = args
		self.cam_name = 2
		self.resize_camera = 0.5
		self.COUNT_SPREADER_IN = 10
		self.net, self.meta = load_model_truck()
		self.door_model = load_model_door()
		# self.working_lane = args.working_lane
		self.video_capture_cam1 = VideoStream(src='rtsp://admin:SPVS@@15411@192.168.2.21/profile3/media.smp').start()
		self.video_capture_cam2 = VideoStream(src='rtsp://admin:SPVS@@15411@192.168.2.22/profile3/media.smp').start()


	def push_door_data(self, result, is_huy=True):
		global door_statistic
		now = datetime.datetime.now()
		folder_save = root_folder_save +'door_data/'

		if not os.path.exists(folder_save):
			os.mkdir(folder_save)

		folder_save = folder_save + str(now.year) + str(now.month) + str(now.day)
		if not os.path.exists(folder_save):
			os.mkdir(folder_save)
		for item in result:
			img_link = ""
			name = str(int(time.time())) + '.jpg'
			door = False
			if item[0] == 0:
				no_door_folder = folder_save +'/'+'Nodoor/'
				door = False
				if not os.path.exists(no_door_folder):
					os.mkdir(no_door_folder)
				img_link = no_door_folder + name
				if is_huy:
					cv2.imwrite(img_link, item[1])
			else:
				door_folder = folder_save +'/'+'Door/'
				door = True
				if not os.path.exists(door_folder):
					os.mkdir(door_folder)
				img_link = door_folder + name
				if is_huy:
					cv2.imwrite(img_link, item[1])
			# print(img_link)
			if is_huy:
				payload = {"data": [
							{
								"image": img_link, 
								"time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
								"door": door
							}
						]
				}
				headers = {'Content-Type': 'application/json'}
				r = requests.post("http://localhost:5000/api/door/result", headers=headers, data = json.dumps(payload))
				print(r.text)

			door_statistic.append(door)

			############################## SENT TO VU ##################
			# if is_huy == False:
			# 	print('sent door to VU')
			# 	result_Vu = {"door": {"is_door": door, "camera_position": "back"}}
			# 	print(result_Vu)
			# 	sio.emit('top_cameras_info', result_Vu)
			############################## SENT TO VU ##################


	def sent_door_to_Vu(self, is_door):
		print(datetime.datetime.now(), 'sent door to VU')
		result_Vu = {"door": {"is_door": is_door, "camera_position": "back"}}
		print(result_Vu)
		sio.emit('top_cameras_info', result_Vu)

	def check_and_sent_door(self):
		global door_statistic, old_value_door
		if len(door_statistic) == 1:
			print(door_statistic)
			self.sent_door_to_Vu(door_statistic[0])
			old_value_door = door_statistic[0]
		elif len(door_statistic) > 1:
			count_true_door = door_statistic.count(True)
			count_false_door = door_statistic.count(False)
			if count_false_door > count_true_door and old_value_door != False:
				old_value_door = False
				print(door_statistic)
				self.sent_door_to_Vu(is_door = False)
			if count_true_door > count_false_door and old_value_door != True:
				old_value_door = True
				print(door_statistic)
				self.sent_door_to_Vu(is_door = True)

	def process_video(self, link_video):
		vidObj = cv2.VideoCapture(link_video) 
		success = 1
		global states, door_statistic, old_value_door, truck_tail_statistic, old_truck_tail_value
		states = {'count_spreader_in':0}
		make_video = True
		out = None

		count = 0
		number_of_frames = vidObj.get(cv2.CAP_PROP_FRAME_COUNT )
		result = None
		current_second = -1
		old_second = -1
		count_no_door = 0
		count_no_truck_tail = 0

		count_truck_tail_in = 0
		count_container_when_truck_tail_in = 0

		while success: 
			# try:
			count += 1
			s = time.time()
			# image = self.video_capture1.read()

			# if current_working_lane in [1,2,3]:
			# 	self.cam_name = 1
			# 	image = self.video_capture_cam1.read()
			# elif current_working_lane in [4,5,6]:
			# 	self.cam_name = 2
			# 	image = self.video_capture_cam2.read()
			time_in_100ms = time.time()*1000/100
			frame_to_read = int(time_in_100ms + 4*30*60)%number_of_frames
			vidObj.set(cv2.CAP_PROP_POS_FRAMES, int(frame_to_read))
			success, image = vidObj.read()
			x1, x2 = 0,0
			x1, x2 = self.crop_lane(current_working_lane)
			image = image[:,x1:x2]

			
			image = cv2.resize(image, None, fx=self.resize_camera, fy=self.resize_camera)

			result = processing_top_view(image, self.net, self.meta, save_folder = False, should_invert = True)
			is_spread = False
			is_truck_tail = False
			now = datetime.datetime.now()
			current_second = now.second
			if current_second % 5 == 0 and old_second != current_second:
				old_second = current_second
				show_cam(image,self.cam_name)
			result_door = []
			for element in result:
				if element['Object'] == 'Container':
					Container_position = element['Position']
					try:
						result_door = processing_door_v2(image, Container_position, self.door_model)
					except:
						print('Exception door detection')

					if count_truck_tail_in > 0:
						count_container_when_truck_tail_in += 1

				if element['Object']  == 'Spreader':
					is_spread = True

				if element['Object'] == 'TruckTail':
					is_truck_tail = True
					print(element)
					count_truck_tail_in += 1

			if count_truck_tail_in == 10:
				if count_container_when_truck_tail_in > 2:
					sio.emit('top_cameras_additional_info', {"truck_empty_when_in": False})
					print({"truck_empty_when_in": False})
				else:
					sio.emit('top_cameras_additional_info', {"truck_empty_when_in": True})
					print({"truck_empty_when_in": True})

			##############################################
			# Process sent truck tail
			
			truck_tail_statistic.append(is_truck_tail)
			self.check_and_sent_truck_tail()
			if is_truck_tail == False:
				count_no_truck_tail += 1
			else:
				count_no_truck_tail = 0

			if count_no_truck_tail == 20:
				count_no_truck_tail = 0
				truck_tail_statistic = []
				old_truck_tail_value = None
				count_truck_tail_in = 0
				count_container_when_truck_tail_in = 0
			# end process sent truck tail

			if is_spread:
				states['count_spreader_in'] += 1
			else:
				states['count_spreader_in'] = 0


			# result_door = []
			try:
				# result_door = processing_door(image, self.net, self.meta, self.door_model, save_folder = False)
				# print(len(result_door))
				if len(result_door) > 0:
					if is_spread and states['count_spreader_in'] == self.COUNT_SPREADER_IN:
						print("push door data to server")
						self.push_door_data(result_door, is_huy = True)
					else:
						self.push_door_data(result_door, is_huy = False)
					self.check_and_sent_door()
				elif len(result_door) == 0:
					count_no_door += 1
				if count_no_door == 10:
					count_no_door = 0
					door_statistic = []
					old_value_door = None

				if len(door_statistic) in [5, 10,20,50]:
					self.sent_door_to_Vu(old_value_door)
			except:
				print('Exception detec door error')
			if count % 100 == 0:
				print("time per frame", time.time() - s)
			if count % 8 ==0:
					print('cccccccccccccccc',count)

	def crop_lane(self, current_lane):
		if current_lane == 5:
			x1, x2 = 600,1500
		elif current_lane == 4:
			x1, x2 = 0,800
		elif current_lane == 6:
			x1, x2 = 1200,1920


		if current_lane == 2:
			x1, x2 = 600,1400
		elif current_lane == 1:
			x1, x2 = 0,800
		elif current_lane == 3:
			x1, x2 = 1200,1920

		return x1, x2

	# def sent_truck_tail_to_Vu()
	def check_and_sent_truck_tail(self):
		global truck_tail_statistic, old_truck_tail_value
		result_Vu = {}
		current_truck_tail_value = None
		if len(truck_tail_statistic) == 1:
			result_Vu = {"truck_tail":truck_tail_statistic[0]}
			sio.emit('top_cameras_info', result_Vu)
			old_truck_tail_value = truck_tail_statistic[0]
			print('Sent truck tail ')
			print(result_Vu)
		elif len(truck_tail_statistic) > 1:
			count_true = truck_tail_statistic.count(True)
			# count_false = truck_tail_statistic.count(False)
			
			if count_true > len(truck_tail_statistic) * 0.2:
				current_truck_tail_value = True
				if old_truck_tail_value != current_truck_tail_value:
					old_truck_tail_value = current_truck_tail_value
					result_Vu = {"truck_tail":old_truck_tail_value}
			else:
				current_truck_tail_value = False
				if old_truck_tail_value != current_truck_tail_value:
					old_truck_tail_value = current_truck_tail_value
					result_Vu = {"truck_tail":old_truck_tail_value}
			if bool(result_Vu):
				print('Sent truck tail ')
				print(result_Vu)
				sio.emit('top_cameras_info', result_Vu)




	def process_streaming(self):
		global states, door_statistic, old_value_door, truck_tail_statistic, old_truck_tail_value
		states = {'count_spreader_in':0}
		make_video = True
		out = None
		count = 0
		result = None
		current_second = -1
		old_second = -1
		count_no_door = 0
		count_no_truck_tail = 0

		count_truck_tail_in = 0
		count_container_when_truck_tail_in = 0

		while True: 
			try:
				count += 1
				s = time.time()
				if current_working_lane in [1,2,3]:
					self.cam_name = 1
					image = self.video_capture_cam1.read()
				elif current_working_lane in [4,5,6]:
					self.cam_name = 2
					image = self.video_capture_cam2.read()
				x1, x2 = self.crop_lane(current_working_lane)
				image = image[:,x1:x2]
				image = cv2.resize(image, None, fx=self.resize_camera, fy=self.resize_camera)

				result = processing_top_view(image, self.net, self.meta, save_folder = False, should_invert = True)
				is_spread = False
				is_truck_tail = False
				now = datetime.datetime.now()
				current_second = now.second
				if current_second % 5 == 0 and old_second != current_second:
					old_second = current_second
					show_cam(image,self.cam_name)

				result_door = []
				for element in result:
					if element['Object'] == 'Container':
						Container_position = element['Position']
						try:
							result_door = processing_door_v2(image, Container_position, self.door_model)
						except:
							print('Exception door detection')
						if count_truck_tail_in > 0:
							count_container_when_truck_tail_in += 1


					if element['Object']  == 'Spreader':
						is_spread = True

					if element['Object'] == 'TruckTail':
						is_truck_tail = True
						count_truck_tail_in += 1

				if count_truck_tail_in == 10:
					if count_container_when_truck_tail_in > 2:
						sio.emit('top_cameras_additional_info', {"truck_empty_when_in": False})
						print({"truck_empty_when_in": False})
					else:
						sio.emit('top_cameras_additional_info', {"truck_empty_when_in": True})
						print({"truck_empty_when_in": True})

				##############################################
				# Process sent truck tail
				
				truck_tail_statistic.append(is_truck_tail)
				self.check_and_sent_truck_tail()
				if is_truck_tail == False:
					count_no_truck_tail += 1
				else:
					count_no_truck_tail = 0
				if count_no_truck_tail == 20:
					count_no_truck_tail = 0
					truck_tail_statistic = []
					old_truck_tail_value = None
					count_truck_tail_in = 0
					count_container_when_truck_tail_in = 0
				# end process sent truck tail


				if is_spread:
					states['count_spreader_in'] += 1
				else:
					states['count_spreader_in'] = 0
				try:
					# result_door = processing_door(image, self.net, self.meta, self.door_model, save_folder = False)
					if len(result_door) > 0:
						if is_spread and states['count_spreader_in'] == self.COUNT_SPREADER_IN:
							print("push door data to server")
							self.push_door_data(result_door, is_huy = True)
						else:
							self.push_door_data(result_door, is_huy = False)
						self.check_and_sent_door()
					elif len(result_door) == 0:
						count_no_door += 1
					if count_no_door == 10:
						count_no_door = 0
						door_statistic = []
						old_value_door = None

					if len(door_statistic) in [5, 10,20,50]:
						self.sent_door_to_Vu(old_value_door)
				except:
					print('Exception detect door error')
					
				if count % 100==0:
					print("time per frame", time.time() - s)
				
			except:
				print("There are something wrong back door")

if __name__ == '__main__':
	args = None
	args = parse_args()
	# link_video = "/data/YRCVideo/Lane5/2019_11_24_13_15_36(LD_20_TW)/cam4.avi"
	# link_video = "/data/YRCVideo/Lane5/2019_11_24_13_39_59(LD_20_TW)/cam4.avi"
	#link_video = "/data/YRCVideo/1202 lane 5/2019_12_2_4_6_48(Lane5)/cam2.avi"
	# link_video = "/data/YRCVideo/folder1/2019_12_13_16_54_46cam2_singlecut.avi"
	
	# link_video = "/media/newhd/new_data/2019_12_8_7_23_53/cam2.avi"
	# link_video = "/media/newhd/new_data/syncron_video/1/2019_12_13_16_54_46cam2.avi"
	# link_video = "/media/newhd/new_data/syncron_video/2/2019_12_13_13_52_6cam2.avi"
	if args.link_video != '':
		Streaming(args).process_video(args.link_video)
	else:
		Streaming(args).process_streaming()
	#python3 Hung_back_door.py --rtsp 'rtsp://admin:dou123789@10.0.0.15:554/cam/realmonitor?channel=1&subtype=0' --resize-camera 0.35 --cam-name 4 --working-lane 5
