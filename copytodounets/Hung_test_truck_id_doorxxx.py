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

from utils_image_video_copy import processing, processing_view, load_model, load_model_ctnr_no, drawCandidate, drawCandidate_top_view, load_model_truck, processing_top_view, processing_door, load_model_door

import json
import requests
import sys
from time import gmtime, strftime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import socketio

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sio = socketio.Client()
sio.connect('http://localhost:8000')
root_folder_save = '/media/newhd/new_data/SPVSResult/QC04/'
# root_folder_save = '/data/Hung/bizlogic/data_save_to_test'


gen = ImageDataGenerator()

list_ids_detected_global = []
states, count_5_frame = None, None

def push_result(result, socket_result):
	try:
		list_result = result
		jsonPayload = {"data": list_result}
		jsonPayload = json.dumps(jsonPayload)
		headers = {'Content-Type': 'application/json'}
		r = requests.post(socket_result, data=jsonPayload, headers=headers)
		return r
	except:
		print('exception push result ')


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
	try:
		frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
		result = {
			"cam": cam_name,
			"image": encode_image64(frame)}

		jsonPayload = json.dumps(result)
		headers = {'Content-Type': 'application/json'}
		r = requests.post("http://localhost:5000/api/home/stream-camera",
						  data=jsonPayload, headers=headers)
	except:
		print('exception show_cam')


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
		'--rtsp', default='rtsp://admin:SPVS@@15411@192.168.2.28/profile3/media.smp', help='rtsp link')
	parser.add_argument('--crop-camera', default='0,1,0.5,1',
						help='crop camera height and width')
	parser.add_argument('--resize-camera', default='0.35',
						help='resize camera height and width')
	parser.add_argument('--cam-name', default='8', help='cam name')

	parser.add_argument('--working-lane', default='4', help='working lane')

	args = parser.parse_args()
	return args


class Streaming(Thread):
	def __init__(self, args):
		Thread.__init__(self)
		self.args = args
		self.video_capture1 = VideoStream(src=args.rtsp).start()
		self.cam_name = args.cam_name
		self.resize_camera = float(args.resize_camera)

		self.COUNT_SPREAD_OUT = 40
		self.COUNT_TRUCK_OUT = 40
		self.MIN_Y_TRUCK_HEAD = 830
		self.COUNT_SPREADER_IN = 1

		# self.COUNT_SPREAD_OUT = 50
		# self.COUNT_TRUCK_OUT = 50
		# self.MIN_Y_TRUCK_HEAD = 800

		self.y_truck_head = - sys.maxsize
		self.net, self.meta = load_model_truck()
		self.door_model = load_model_door()
		self.working_lane = args.working_lane

		# self.crop_lane_x1,self.crop_lane_x2 = None, None
		if self.working_lane == '5':
			self.crop_lane_x1,self.crop_lane_x2 = 1000,2700
		elif self.working_lane =='6':
			self.crop_lane_x1,self.crop_lane_x2 = 0,1700
		elif self.working_lane =='4':
			self.crop_lane_x1,self.crop_lane_x2 = 2000,3840


		if self.working_lane == '2':
			self.crop_lane_x1,self.crop_lane_x2 = 1000,2700
		elif self.working_lane =='3':
			self.crop_lane_x1,self.crop_lane_x2 = 0,1700
		elif self.working_lane =='1':
			self.crop_lane_x1,self.crop_lane_x2 = 2000,3840

	def default_state(self):
		global states, count_5_frame
		states = {'truck_Id': None,'truck_Id_cof': 0,'count_truck_out': 0,'count_spread_out': 0, 'waiting_spreader_out': False, 'count_spreader_in': 0}
		count_5_frame = {'container':[], 'truck_id':[], 'spreader':[]}
		# states = {'no_truck': False,'truck_empty':False, 'spreader_on_camera': False, 'truck_id': None,'truck_Id_conf': 0, 'count_current_state': 0, 'initial_container': None}

	def check_sent_result(self):
		is_spreader_on_camera = False
		
		if sum(count_5_frame['spreader']) >=3:
			is_spreader_on_camera = True
			
		is_truck_id = False
		if sum (count_5_frame['truck_id']) >=3:
			is_truck_id = True
		
		if  is_truck_id == False:
			states['truck_Id'] = None
			states['truck_Id_cof'] = 0
			
		is_container = False
		if sum(count_5_frame['container']) >=3:
			is_container = True
			
		if is_truck_id == False:
			is_container = False
		result = {'spreader_on_camera':is_spreader_on_camera, 'no_truck': not is_truck_id, 'truck_id':states['truck_Id'], 'truck_empty': not is_container}
		return result

	def check_change(self, result, before_result):
		count = 0
		for k, v in result.items():
			if result[k] != before_result[k]:
				count += 1
				break
		return count
	def push_door_data(self, result):
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
				cv2.imwrite(img_link, item[1])
			else:
				door_folder = folder_save +'/'+'Door/'
				door = True
				if not os.path.exists(door_folder):
					os.mkdir(door_folder)
				img_link = door_folder + name
				cv2.imwrite(img_link, item[1])
			print(img_link)
			payload = {"data": [
						{
							"image": img_link, 
							"time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
							"door": door
						}
					]
			}
			headers = {'Content-Type': 'application/json'}
			try:
				r = requests.post("http://localhost:5000/api/door/result", headers=headers, data = json.dumps(payload))
				print(r.text)
			except:
				print('exception push_door_data')

			############################### SENT TO VU ##################
			print('sent door to VU')
			result_Vu = {"door": {"is_door": door, "camera_position": "front"}}
			print(result_Vu)
			sio.emit('top_cameras_info', result_Vu)
			############################### SENT TO VU ##################

	def push_truck_id(self, image, truck_id):
		now = datetime.datetime.now()
		folder_save = root_folder_save +'truck_id_data/'
		if not os.path.exists(folder_save):
			os.mkdir(folder_save)
			
		folder_save = folder_save + str(now.year) + str(now.month) + str(now.day)
		if not os.path.exists(folder_save):
			os.mkdir(folder_save)
		img_link = folder_save +'/'+ str(int(time.time())) + '.jpg'
		image = gen.apply_transform(image, {'theta': 180})
		cv2.imwrite(img_link, image)
		
		payload = {
					"data": [
						{
					"image": img_link, 
					"truck_id": truck_id, 
					"time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
					"cam_num": self.cam_name,
					"lane": self.working_lane
						}]
				}
		headers = {'Content-Type': 'application/json'}
		try:
			r = requests.post("http://localhost:5000/api/itv/result", headers=headers, data = json.dumps(payload))
			print(payload)
			print(r.text)
		except:
			print('exception push truck_id')
			
	def process_video(self, link_video):
		vidObj = cv2.VideoCapture(link_video) 
		success = 1
		self.default_state()
		print(states)
		sio.emit('top_cameras_info', self.check_sent_result())
		make_video = False
		out = None
		if make_video:
			frame_width = int(vidObj.get(3))
			frame_height = int(vidObj.get(4))
			print(frame_width, frame_height)
			font = cv2.FONT_HERSHEY_SIMPLEX 
			org = (0, 185) 
			fontScale = 1
			color = (0, 0, 255) 
			thickness = 2
			fourcc = cv2.VideoWriter_fourcc(*'XVID') 
			out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(frame_width*0.5), int(frame_height*0.5))) 
		count = 0
		before_result = None
		result = None
		before_truck_id = None
		truck_head_position = None
		is_container_before = False
		#################################
		print('set specific fram')
		vidObj.set(1, 5*60*30 + 2*60*30)
		####################################
		while success: 
			count += 1
			# success, image = vidObj.read()
			success, image = vidObj.read()
			s = time.time()
			if success:
				image = image[:,self.crop_lane_x1:self.crop_lane_x2]
				image = cv2.resize(image,None,fx=0.5,fy=0.5)
				result = processing_top_view(image, self.net, self.meta, save_folder = False, should_invert = True)
				
				test_result = result
				is_spread = False
				is_truck = False
				is_container = False

				if count % 10 == 0:
					show_cam(image,self.cam_name)
					if count == 10000000:
						count = 0
				for element in result:
					if element['Object'] == 'Container':
						is_container = True
						
					if element['Object'] == 'TruckHead':
						is_truck = True
						self.y_truck_head = element['Position'][1]
						
						truck_head_position = element['Position']
						x,y,w,h = truck_head_position
						image_truck_head = image[y-h//2: y+h//2, x-w//2:x+w//2]
						if element['Truck ID'][1] != 1 and element['Truck ID'][1] > states['truck_Id_cof'] and len(element['Truck ID'][0]) ==3:
							states['truck_Id_cof'] = element['Truck ID'][1]
							states['truck_Id'] =  element['Truck ID'][0]
					if element['Object']  == 'Spreader':
						is_spread = True
						states['waiting_spreader_out'] = True

				if is_spread == False:
					states['waiting_spreader_out']

				if is_spread:
					states['count_spreader_in'] += 1
				else:
					states['count_spreader_in'] = 0

				if states['count_spread_out'] and is_truck == False:
					states['count_truck_out'] += 1
				if len(count_5_frame['container'])==5:
					count_5_frame['container'].pop(0)
					count_5_frame['truck_id'].pop(0)
					count_5_frame['spreader'].pop(0)
				count_5_frame['container'].append(is_container)
				count_5_frame['truck_id'].append(is_truck)
				count_5_frame['spreader'].append(is_spread)

				if states['waiting_spreader_out'] and is_spread == False:
					states['count_spread_out'] += 1

				if is_spread and states['count_spreader_in'] == self.COUNT_SPREADER_IN:
					result_door = processing_door(image, self.net, self.meta, self.door_model, save_folder = False)
					print('result_door')
					if len(result_door) > 0:
						print("push door data to server")
						self.push_door_data(result_door)

				if states['count_spread_out'] == self.COUNT_SPREAD_OUT and states['count_truck_out'] == self.COUNT_TRUCK_OUT and self.y_truck_head >= self.MIN_Y_TRUCK_HEAD:
					print('Reset default state')
					self.default_state()
				result = self.check_sent_result()
				if before_result == None:
					before_result = result  
				
				# print(test_result)
				if self.check_change(result, before_result) and self.y_truck_head >= self.MIN_Y_TRUCK_HEAD:
					before_result = result
					############################### SENT TO VU ##################
					print('sent to VU')
					result['truck_image'] = 'test.jpg'
					sio.emit('top_cameras_info', result)
					############################### SENT TO VU ##################
					if make_video:
						for i in range(10):
							image = cv2.putText(image, str(result), org, font, fontScale, color, thickness, cv2.LINE_AA, False) 
							out.write(image)
					print("----------------sent result ------------------")
					print(result)
					print("----------------sent result ------------------")

					if before_truck_id != states['truck_Id'] and states['truck_Id'] != None and  states['truck_Id'] != '':
						before_truck_id = states['truck_Id']
						if truck_head_position != None:
							print("push_truck_id")
							self.push_truck_id(image_truck_head, states['truck_Id'])
					# if is_container_before != is_container:
					# 	is_container_before = is_container
					# 	try:
					# 		result = processing_door(image, self.net, self.meta, self.door_model, save_folder = False)
					# 	except:
					# 		result = []
					# 	if len(result) > 0:
					# 		print("push door data to server")
					# 		self.push_door_data(result)
				if make_video:
					out.write(image)
			else:
				break
		vidObj.release()
		if make_video:
			out.release()    

	def process_streaming(self):
		self.default_state()
		print(states)
		sio.emit('top_cameras_info', self.check_sent_result())
		count = 0
		before_result = None
		result = None
		before_truck_id = None
		truck_head_position = None
		is_container_before = False


		while True: 
			# try:
			count += 1
			s = time.time()
			image = self.video_capture1.read()
			# print(image.shape)
			# print('BBBB')
			image = image[:,self.crop_lane_x1:self.crop_lane_x2]
			image = cv2.resize(image, None, fx=self.resize_camera, fy=self.resize_camera)
			result = processing_top_view(image, self.net, self.meta, save_folder = False, should_invert = True)
			
			test_result = result
			is_spread = False
			is_truck = False
			is_container = False

			if count % 100 == 0:
				show_cam(image,self.cam_name)
				if count == 10000000:
					count = 0
			for element in result:
				if element['Object'] == 'Container':
					is_container = True
					
				if element['Object'] == 'TruckHead':
					is_truck = True
					self.y_truck_head = element['Position'][1]
					
					truck_head_position = element['Position']
					x,y,w,h = truck_head_position
					image_truck_head = image[y-h//2: y+h//2, x-w//2:x+w//2]
					if element['Truck ID'][1] != 1 and element['Truck ID'][1] > states['truck_Id_cof'] and len(element['Truck ID'][0]) == 3:
						states['truck_Id_cof'] = element['Truck ID'][1]
						states['truck_Id'] =  element['Truck ID'][0]
				if element['Object']  == 'Spreader':
					is_spread = True
					states['waiting_spreader_out'] = True

			if is_spread == False:
				states['waiting_spreader_out']

			if is_spread:
				states['count_spreader_in'] += 1
			else:
				states['count_spreader_in'] = 0

			if states['count_spread_out'] and is_truck == False:
				states['count_truck_out'] += 1
			if len(count_5_frame['container'])==5:
				count_5_frame['container'].pop(0)
				count_5_frame['truck_id'].pop(0)
				count_5_frame['spreader'].pop(0)
			count_5_frame['container'].append(is_container)
			count_5_frame['truck_id'].append(is_truck)
			count_5_frame['spreader'].append(is_spread)

			if states['waiting_spreader_out'] and is_spread == False:
				states['count_spread_out'] += 1

			if is_spread and states['count_spreader_in'] == self.COUNT_SPREADER_IN:
				result_door = processing_door(image, self.net, self.meta, self.door_model, save_folder = False)
				print('result_door')
				if len(result_door) > 0:
					print("push door data to server")
					self.push_door_data(result_door)
			
			if states['count_spread_out'] == self.COUNT_SPREAD_OUT and states['count_truck_out'] == self.COUNT_TRUCK_OUT and self.y_truck_head >= self.MIN_Y_TRUCK_HEAD:
				print('Reset default state')
				self.default_state()
			result = self.check_sent_result()
			if before_result == None:
				before_result = result  
			
			# print(test_result)
			if self.check_change(result, before_result) and self.y_truck_head >= self.MIN_Y_TRUCK_HEAD:
				before_result = result
				############################### SENT TO VU ##################
				print('sent to VU')
				result['truck_image'] = 'test.jpg'
				sio.emit('top_cameras_info', result)
				############################### SENT TO VU ##################
				
				print("----------------sent result ------------------")
				print(result)
				print("----------------sent result ------------------")

				if before_truck_id != states['truck_Id'] and  states['truck_Id'] != '':
					before_truck_id = states['truck_Id']
					if truck_head_position != None:
						print("push_truck_id")
						self.push_truck_id(image_truck_head, states['truck_Id'])
					# try:
					# 	result = processing_door(image, self.net, self.meta, self.door_model, save_folder = False)
					# except:
					# 	print('exeption door')
					# 	result = []
					# if len(result) > 0:
					# 	print("push door data to server")
					# 	self.push_door_data(result)
				# if is_container_before != is_container:
				# 	is_container_before = is_container
				# 	try:
				# 		result = processing_door(image, self.net, self.meta, self.door_model, save_folder = False)
				# 	except:
				# 		print('exeption door')
				# 		result = []
				# 	if len(result) > 0:
				# 		print("push door data to server")
				# 		self.push_door_data(result)
			# except:
			# 	print('There are something wrong')


if __name__ == '__main__':
	# args = None
	args = parse_args()
	# link_video = "/data/YRCVideo/Lane5/2019_11_24_13_15_36(LD_20_TW)/cam4.avi"
	# link_video = "/data/YRCVideo/Lane5/2019_11_24_13_39_59(LD_20_TW)/cam4.avi"
	#link_video = "/data/YRCVideo/1202 lane 5/2019_12_2_4_6_48(Lane5)/cam4.avi"
	# link_video = "/data/YRCVideo/Lane5/2019_11_24_6_17_38(DS_20_TW)/cam4.avi"

	link_video = "/media/newhd/new_data/2019_12_8_7_23_53/cam4.avi"
	# link_video = "/media/newhd/new_data/2019_12_9_7_50_24/cam4.avi"

	Streaming(args).process_video(link_video)
	#while True:
	#	print('start stream')
	#	Streaming(args).process_streaming()

	#python3 Hung_test_truck_id_door.py --rtsp 'rtsp://admin:dou123789@10.0.0.15:554/cam/realmonitor?channel=1&subtype=0' --resize-camera 0.35 --cam-name 4 --working-lane 5
