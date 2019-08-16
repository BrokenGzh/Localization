# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:00:34 2019

@author: GZI1SGH
"""

# coding: utf-8
from imageai.Detection import VideoObjectDetection
import os

def forFrame(frame_number, output_array, output_count):
    print("FOR FRAME " , frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
    print("------------END OF A FRAME --------------")
 
execution_path=os.getcwd()  

detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(
    os.path.join(execution_path,'resnet50_coco_best_v2.0.1.h5'))
detector.loadModel(detection_speed="fast")

custom_objects = detector.CustomObjects(person=True, bicycle=True, motorcycle=True)

custom_objects = detector.CustomObjects(
    person=True, bicycle=True, motorcycle=True) 

video_path = detector.detectCustomObjectsFromVideo(
    custom_objects=custom_objects,
    input_file_path=os.path.join(execution_path, 'traffic.mp4'),
    output_file_path=os.path.join(execution_path,  'traffic-detected'),
    frames_per_second=20,
    frame_detection_interval=5,
    per_frame_function=forFrame,
    minimum_percentage_probability=30)