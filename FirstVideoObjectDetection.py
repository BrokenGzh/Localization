# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:00:34 2019

@author: GZI1SGH
"""

# coding: utf-8
from imageai.Detection import VideoObjectDetection
import os
import math
import numpy as np

def estimateSpeed(location1, location2):
	d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
	# ppm = location2[2] / carWidht
	ppm = 8.8
	d_meters = d_pixels / ppm
	#print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
	fps = 18
	speed = d_meters * fps * 3.6
	return speed
 
def estimateangle(location1,location2):
    if (location2[0] - location1[0])==0:
        if(location2[1] - location1[1])<0:
            return 90.0
        if(location2[1] - location1[1])>0:
            return 270.0
        if(location2[1] - location1[1])==0:
            return 0.0
    else:
        if (location2[0] - location1[0])<0:
            ang=math.pi-math.atan((location2[1]-location1[1])/(location2[0]-location1[0]))
        else:
            if(location2[1]-location1[1])>0:
                ang=2*math.pi-math.atan((location2[1]-location1[1])/(location2[0]-location1[0]))
            else:
                ang=-math.atan((location2[1]-location1[1])/(location2[0]-location1[0]))
        return ang/math.pi*180
        
         
 
past=np.zeros([10,4])
current=np.zeros([10,4])
location1=[0,0]
location2=[0,0]
velocity1=[0.0]*10
velocity2=[0.0]*10
acceleration=[0.0]*10

#def forFrame(frame_number, output_array, output_count):
#    print("FOR FRAME " , frame_number)
#    print("Output for each object : ", output_array)
#    print("Output count for unique objects : ", output_count)
#    print("------------END OF A FRAME --------------")

def forFrame(frame_number, output_array, output_count):
    i=0
    for data in output_array:
        past[i]=current[i]
        current[i]=data["box_points"]
#        print(current[i][0],current[i][2])
#    print(len(output_array))
#    print("---")
#            
        location1[0]=(past[i][0]+past[i][2])/2
        location1[1]=(past[i][1]+past[i][3])/2
        location2[0]=(current[i][0]+current[i][2])/2
        location2[1]=(current[i][1]+current[i][3])/2
#        print(i,'\t',location2)
#        velocity1[i]=velocity2[i]
#        velocity2[i]=estimateSpeed(location1, location2)
        heading=estimateangle(location1,location2)
        print(i,":angle",heading)
        
#        print(velocity2[i])
#        acceleration[i]=(velocity2[i]-velocity1[i])*18/8.8
#        print(acceleration)
##        print("velocity:",velocity2[i])
##        print("acceleration",acceleration[i])
        i+=1
#            
#==============================================================================
#             location[i]=data["box_points"]
#     for i in range(len(output_array)):
#         print(location[i])
#==============================================================================
        
execution_path=os.getcwd()  

detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(
    os.path.join(execution_path,'resnet50_coco_best_v2.0.1.h5'))
detector.loadModel(detection_speed="fast")

custom_objects = detector.CustomObjects(car=True)

video_path = detector.detectCustomObjectsFromVideo(
    custom_objects=custom_objects,
    input_file_path=os.path.join(execution_path, 'traffic.mp4'),
    output_file_path=os.path.join(execution_path,  'traffic-detected'),
    frames_per_second=20,
#==============================================================================
#     frame_detection_interval=1,
#==============================================================================
    per_frame_function=forFrame,
    minimum_percentage_probability=30)