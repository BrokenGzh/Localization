# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:00:34 2019

@author: GZI1SGH
"""

# coding: utf-8
from imageai.Detection import VideoObjectDetection
import os
 
execution_path=os.getcwd()  
 
detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(
    os.path.join(execution_path,'resnet50_coco_best_v2.0.1.h5'))
detector.loadModel()

custom_objects = detector.CustomObjects(
    person=True, bicycle=True, motorcycle=True) 

video_path = detector.detectObjectsFromVideo(
    input_file_path=os.path.join(execution_path, 'traffic.mp4'),
    output_file_path=os.path.join(execution_path, 'videos',
                                  'traffic-detected'),
    frames_per_second=20,
    log_progress=True)
print(video_path)