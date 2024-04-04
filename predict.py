import cv2
import serial
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time
from collections import defaultdict
import matplotlib.pyplot as plt

# Serial setting
ser = serial.Serial('/dev/ttyUSB0',115200)
# 카메라 프레임의 원하는 너비와 높이를 정의합니다.
W, H = 640, 480

# RealSense 카메라 파이프라인 초기화
config = rs.config()
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

pipeline = rs.pipeline()
profile = pipeline.start(config)

# 컬러와 깊이 이미지 스트림의 정렬을 설정합니다.
align_to = rs.stream.color
align = rs.align(align_to)

def calculate_angle(a,b,c):
    
    # 각 값을 받아 넘파이 배열로 변형
    a = np.array(a) # 첫번째
    b = np.array(b) # 두번째
    c = np.array(c) # 세번째

    # 라디안을 계산하고 실제 각도로 변경한다.
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    # 180도가 넘으면 360에서 뺀 값을 계산한다.
    if angle >180.0:
        angle = 360-angle

    # 각도를 리턴한다.
    return angle

# YOLOv8 모델을 로드합니다.
model_pose = YOLO("yolov8m-pose")
model_hands = YOLO("bestv2.pt")

#Find hands.
object_name = 'none'
# Initialize variables outside of your main processing loop
prev_cx_stop, prev_cy_stop, prev_cx_move, prev_cy_move = None, None, None, None  # Previous center coordinates
current_cx_stop, current_cy_stop, current_cx_move, current_cy_move = None, None, None, None
change_threshold = 15  # Threshold for detecting significant change
angle =0

while True:
    time1 = time.time()
    frames = pipeline.wait_for_frames()  # RealSense로부터 컬러 및 깊이 이미지 프레임을 검색합니다.

    aligned_frames = align.process(frames)  # 깊이 프레임을 컬러 프레임의 관점으로 정렬합니다.
    color_frame = aligned_frames.get_color_frame()
    if not color_frame:
        continue  # 컬러 이미지 데이터가 없으면 프레임을 건너뜁니다.

    color_image = np.asanyarray(color_frame.get_data())  # 프레임 데이터를 NumPy 배열로 변환합니다.

    results_hands = model_hands(color_image, conf=0.8, verbose=False)  # YOLOv8 모델을 사용하여 컬러 이미지에서 탐지 결과를 얻습니다.
    
    if results_hands is not None:
        for r in results_hands:
            boxes = r.boxes
            for box in boxes:
                class_index = box.cls  # Get the class index of the object
                object_name = model_hands.names[int(class_index)]  # Use the index to get the object's name
                if object_name == 'Stop':
                    # Calculate current center coordinates
                    b = box.xyxy[0].to('cpu').detach().numpy().copy()  # Move results from GPU to CPU
                    c = box.cls
                    x1, y1, x2, y2 = map(int, b[:4])
                    current_cx_stop, current_cy_stop = int((x2 - x1) / 2 + x1), int((y2 - y1) / 2 + y1)
                    
                    if prev_cx_stop is not None and prev_cy_stop is not None:
                    # Calculate Euclidean distance between previous and current center
                        distance = np.sqrt((current_cx_stop - prev_cx_stop)**2 + (current_cy_stop - prev_cy_stop)**2)
                        # print(distance)
                    
                        if distance > change_threshold:
                            # print("Release")
                            object_name='R'
                        else :
                            # print("Stop")
                            object_name='S'
                            
                    prev_cx_stop, prev_cy_stop = current_cx_stop, current_cy_stop      

                    # Draw bounding box and center annotation on the image
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 0, 255), thickness=2, lineType=cv2.LINE_4)
                    cv2.putText(color_image,  model_hands.names[int(c)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, (0, 0, 255), 2, cv2.LINE_4)
                elif object_name == 'Move on':
                    # Calculate current center coordinates
                    b = box.xyxy[0].to('cpu').detach().numpy().copy()  # Move results from GPU to CPU
                    c = box.cls
                    x1, y1, x2, y2 = map(int, b[:4])
                    current_cx_move, current_cy_move = int((x2 - x1) / 2 + x1), int((y2 - y1) / 2 + y1)
                    
                    prev_cx_move, prev_cy_move = current_cx_move, current_cy_move
                    object_name='M'
                    print(angle)
                    # Draw bounding box and center annotation on the image
                    # print(current_cx_move,current_cy_move)
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 0, 255), thickness=2, lineType=cv2.LINE_4)
                    cv2.putText(color_image,  model_hands.names[int(c)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, (0, 0, 255), 2, cv2.LINE_4)
                elif object_name == 'You':
                    b = box.xyxy[0].to('cpu').detach().numpy().copy()  # Move results from GPU to CPU
                    c = box.cls
                    x1, y1, x2, y2 = map(int, b[:4])
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 0, 255), thickness=2, lineType=cv2.LINE_4)
                    cv2.putText(color_image,  model_hands.names[int(c)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, (0, 0, 255), 2, cv2.LINE_4)
                    object_name='Y'
                    print(object_name)
                else:
                    b = box.xyxy[0].to('cpu').detach().numpy().copy()  # Move results from GPU to CPU
                    c = box.cls
                    x1, y1, x2, y2 = map(int, b[:4])
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 0, 255), thickness=2, lineType=cv2.LINE_4)
                    cv2.putText(color_image,  model_hands.names[int(c)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, (0, 0, 255), 2, cv2.LINE_4)
                
                # print(object_name, angle)
                                              
    #inferencing        
    results_pose = model_pose(color_image, conf=0.8, verbose=False)

    pose_color_image = results_pose[0].plot()
    #Find pose.
    pointcount = 6
    skeleton_point= np.zeros((pointcount,2))
    if results_pose is not None:
        for r in results_pose:
            keypoints = r.keypoints
            # print(keypoints.xy)  # 객체의 이름을 출력합니다.
            for i, k in enumerate(keypoints):
                if k.xy[0].size(0) > 6:  # Ensure there are enough elements
                    skeleton_point[0] = k.xy[0][6].cpu().numpy()  # Right shoulder
                if k.xy[0].size(0) > 8:  # Ensure there are enough elements
                    skeleton_point[1] = k.xy[0][8].cpu().numpy()  # Right elbow
                if k.xy[0].size(0) > 10:  # Ensure there are enough elements
                    skeleton_point[2] = k.xy[0][10].cpu().numpy()  # Right wrist
                    angle=calculate_angle(skeleton_point[0],skeleton_point[1],skeleton_point[2])

            # for i,k in enumerate(keypoints):
            #     skeleton_point[0]=k.xy[0][6].cpu().numpy() #Right shoulder
            #     # r_sho=k.xy[0][6].cpu().numpy() #Right shoulder
            #     skeleton_point[1]=k.xy[0][8].cpu().numpy() #Right shoulder
            #     # r_elb=k.xy[0][8].cpu().numpy() #Right shoulder
            #     skeleton_point[2]=k.xy[0][10].cpu().numpy() #Right shoulder
            #     # r_win=k.xy[0][10].cpu().numpy() #Right shoulder
            #     angle=calculate_angle(skeleton_point[0],skeleton_point[1],skeleton_point[2])
            #     # print(angle)
    
    conditions = {
        "Forward": lambda angle: angle > 120 and angle < 150,
        "Backward": lambda angle: angle > 90 and angle <110,
        "Turn": lambda angle: angle>0 and angle < 60,
        # "Move on": lambda angle: angle > 150,
    }
    if object_name=='Forward':
        object_name='F'
    elif object_name=='Backward':
        object_name='B'
    elif object_name=='Turn':
        object_name='T'
    
    # if conditions.get(object_name, lambda x: False)(angle):
    #     print(object_name,angle)
    ser.write(str(object_name).encode('utf-8')+b'\r\n')
    cv2.imshow("color_image", pose_color_image)  # 주석 처리된 부분은 필요에 따라 활성화할 수 있습니다.
    
    # cv2.imshow("color_image", color_image)  # 주석 처리된 부분은 필요에 따라 활성화할 수 있습니다.
    time2 = time.time()
    # print(f"FPS : {1 / (time2 - time1):.2f}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        ser.close()

pipeline.stop()  # 카메라 파이프라인을 종료합니다.
