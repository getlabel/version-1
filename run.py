# USAGE
# python run.py --image_folder car/ 
import argparse
import time
import cv2
import os
from utiles import YoloV3

ap = argparse.ArgumentParser()

ap.add_argument("-i","--image_folder", required=True,
	help="path to input image folder")
args = vars(ap.parse_args())

counter=0

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 

images_path=os.path.sep.join([ROOT_DIR, args['image_folder']])

list_images=os.listdir(images_path)

yolo_obj=YoloV3(images_path)

net=yolo_obj.load_weights()

output_directory = os.path.join(ROOT_DIR, "xmls")

if not os.path.isdir(output_directory):
	os.mkdir(output_directory)

for image in list_images:
	yolo_obj.process_images(image,net)
	yolo_obj.generate_result(image,counter, args['image_folder'])
	counter+=1






