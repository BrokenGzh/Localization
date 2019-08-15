# coding: utf-8
from imageai.Detection import ObjectDetection
import os
 
execution_path=os.getcwd()  # imageAI根目录
 
detector = ObjectDetection()  # 加载检测器
 
#resnet png格式可以正常输出文本，且有标记的新图片
detector.setModelTypeAsRetinaNet()  # 设置模型网络类型
detector.setModelPath(
os.path.join(execution_path,
                "resnet50_coco_best_v2.0.1.h5"))  # 设置模型文件路径
 
# YOLOv3 用不了
# detector.setModelTypeAsYOLOv3()
# detector.setModelPath(
#     os.path.join(execution_path, 'models', 'Object Detection', 'yolo.h5'))
 
# TinyYOLOv3 png格式可以正常输出文本，且有标记的新图片
# =============================================================================
# detector.setModelTypeAsTinyYOLOv3()
# detector.setModelPath(
#     os.path.join(execution_path, 'models', 'Object Detection',
#                  "yolo-tiny.h5"))
# =============================================================================
 
detector.loadModel()  # 加载模型
detections = detector.detectObjectsFromImage(
    input_image=os.path.join(execution_path,
                             "image2.jpg"),  # 输入待检测图片路劲
    output_image_path=os.path.join(execution_path,
                                   "image2new.png"),  # 输出图片路径 特别提醒用png别用jpg
    minimum_percentage_probability=30)  # 输出检测到的物品的最小可能性阈值
 
for eachObject in detections:
    print(eachObject["name"], " : ", eachObject["percentage_probability"],
          ' : ', eachObject.get("box_points"))  # 目标名 : 可能性大小 : 目标区域
    print("--------------------------------")