"""utiles.py called by run.py"""
import numpy as np
import time
import cv2
import os

class YoloV3():
	
	"""Yolo class the whole detection is peforming here."""
	
	def __init__(self,images_path):
		self.images_path=images_path
		self.boxes = list()
		self.confidences = list()
		self.classIDs = list()
		self.full_path = list()
		self.height=""
		self.width=""

	def load_weights(self):
		
		"""Loading the yolo weights and the config files"""		
		
		weightsPath = os.path.sep.join(["weights", "yolov3.weights"])
		configPath = os.path.sep.join(["weights", "yolov3.cfg"])		
		net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
		return net
	
	def process_images(self,image,net):
		
		"""The whole detection part is pefroming here and also the bounding box values
			labels and confidence values."""

		image=os.path.sep.join([self.images_path,image])		
		img = cv2.imread(image)
		(H, W) = img.shape[:2]
		self.height=H
		self.width=W				
		ln = net.getLayerNames()
		ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]		
		blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
			swapRB=True, crop=False)
		net.setInput(blob)
		start = time.time()
		layerOutputs = net.forward(ln)
		end = time.time()
		
		print("YOLO took {:.6f} seconds".format(end - start))
		self.full_path.append(image)

		for output in layerOutputs:			

			for detection in output:				
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]				

				if confidence > 0.6:					
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					self.boxes.append([x, y, int(width), int(height)])
					self.confidences.append(float(confidence))
					self.classIDs.append(classID)

	def generate_result(self,image_name,counter,folder):

		"""Generating the xml files for every image we have for the labeling as a result,
		we get the xml files with complete coordinate values."""
		
		get_name=image_name.split(".")
		get_name.pop()
		get_name.insert(1,"xml")
		xml_file_name=".".join(get_name)

		labels_path = os.path.sep.join(["weights", "coco.names"])
		labels = open(labels_path).read().strip().split("\n")

		xmin=self.boxes[counter][0]
		ymin=self.boxes[counter][1]
		xmax=self.boxes[counter][2]+xmin
		ymax=self.boxes[counter][3]+ymin

		xml_content = open("base.xml").read()
		replacement_dict={"[filename]":image_name,"[image_path]":self.full_path[counter],
						"[label]":labels[self.classIDs[counter]],"[width]":self.width,"[height]":self.height,
						"[xmin]":xmin,"[ymin]":ymin,"[xmax]":xmax,
						"[ymax]":ymax,"[folder]":folder.strip("/")}					
		for x,y in replacement_dict.items():			
			xml_content=xml_content.replace(x,str(y))		

		write_content = open("xmls/{}".format(xml_file_name),"w")
		write_content.write(xml_content)



