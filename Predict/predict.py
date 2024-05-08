#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import time
import serial
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Serial setting
# ser = serial.Serial('/dev/ttyUSB0', 115200)

# Define camera frame width,height
W, H = 640, 480

# Init RealSense pipeline
config = rs.config()
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)

pipeline = rs.pipeline()
profile = pipeline.start(config)

# Set image stream align
align_to = rs.stream.color
align = rs.align(align_to)

gesture = None
count_print = 0

def calculate_angle_arm(a, b, c):

    # 각 값을 받아 넘파이 배열로 변형
    a = np.array(a)  # 첫번째
    b = np.array(b)  # 두번째
    c = np.array(c)  # 세번째

    # 라디안을 계산하고 실제 각도로 변경한다.
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
        np.arctan2(a[1]-b[1], a[0]-b[0])
    angle_arm = np.abs(radians*180.0/np.pi)

    # 180도가 넘으면 360에서 뺀 값을 계산한다.
    if angle_arm > 180.0:
        angle_arm = 360-angle_arm

    # 각도를 리턴한다.
    return angle_arm

#Road YOLO model
model_pose = YOLO("yolov8m-pose")
model_hands = YOLO("240503.pt")

box_cx, box_cy = None, None  # predict box
pbox_cx, pbox_cy = None, None  # pointing box

# Previous center coordinates
pre_cx_stop, pre_cy_stop, pre_cx_pointing, pre_cy_pointing = None, None, None, None
cur_cx_stop, cur_cy_stop, cur_cx_pointing, cur_cy_pointing = None, None, None, None
threshold_waving = 40  # Threshold for waving

angle_arm = 0

count_gesture=0
gesture_pre='N'

while True:
    # time1 = time.time() #for measure FPS
    frames = pipeline.wait_for_frames()  # Searching color and depth image from RealSense
    aligned_frames = align.process(frames)  # Aling depth frame from color frame aspect
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if not color_frame: # Skip frame dosen't exigst color image data
        continue  

    # Transform frame data to Numpy array
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    
    results_hands = model_hands(
        color_image, conf=0.8, verbose=False)  # Predict hands
    
    hands = 'N' #Init hands
    final_hands ='N'
    
    if results_hands is not None:
        
        pre_cx_stop = None
        pre_cy_stop = None
        
        for r in results_hands:
            boxes = r.boxes #Boxes class
            depth_box = [0] * (len(boxes))
            multi_hands = [''] * len(boxes)
            min_depth = float('inf')
            final_hands_index=0
            
            for index,box in enumerate(boxes): #Multi box detect
                b = box.xyxy[0].to('cpu').detach().numpy().copy()
                x1, y1, x2, y2 = map(int, b[:4]) #Box left top and right bottom coordinate
                box_cx, box_cy = int(
                    (x2 - x1) / 2 + x1), int((y2 - y1) / 2 + y1) #Get box center coordinate
                depth = depth_frame.get_distance(box_cx, box_cy)
                depth_box[index] = depth
                multi_hands[index] = model_hands.names[int(box.cls)]

                # Drawing bounding box
                if depth < min_depth:
                    min_depth = depth
                    final_hands_index = index
                
                cv2.rectangle(color_image, (x1, y1), (x2, y2),
                                  (0, 0, 255), thickness=2, lineType=cv2.LINE_4)
                cv2.putText(color_image, f"Depth: {depth}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(color_image, multi_hands[index], (box_cx, box_cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            if len(multi_hands) > final_hands_index:  # Ensure index is within bounds
                final_hands = multi_hands[final_hands_index]
                # print(f"Closest Hand Gesture: {final_hands} at Index {final_hands_index} with Depth {min_depth}")
                
                if final_hands == 'STOP':
                    cur_cx_stop, cur_cy_stop = int(
                        (x2 - x1) / 2 + x1), int((y2 - y1) / 2 + y1) #Set current box coordinate
                    if pre_cx_stop is not None and pre_cy_stop is not None:
                        distance_stop=abs(cur_cx_stop - pre_cx_stop)#Get distance
                        print(distance_stop)
                        multi_hands[index] = 'W' if distance_stop > threshold_waving else 'S'
                        
                        if distance_stop > threshold_waving: #Check sharply movement using only x value
                            multi_hands[index] = 'W'

                    pre_cx_stop, pre_cy_stop = cur_cx_stop, cur_cy_stop
                    
                elif final_hands == 'POINTING':
                    multi_hands[index] = 'P'
                    pbox_cx, pbox_cy = box_cx, box_cy
                else:
                    multi_hands[index]=final_hands[0]
                

    results_pose = model_pose(color_image, conf=0.8, verbose=False) # Predict pose
    pose_color_image = results_pose[0].plot() #Draw skelton to pose image
    
    count_point = 7  # Number of array for pose coordinate 
    
    distance_whl, distance_whr = None, None # Distance both side winkle-hands
    value_slx, value_srx = None, None # Value shoulder x
    
    depth_nose=None

    coordinate_pose = np.zeros((count_point, 2)) #[RS,RE,RW,LS,LE,LW,N]
    if results_pose is not None:
        for r in results_pose:
            keypoints = r.keypoints
            pose_boxes = r.boxes
            if len(pose_boxes.xyxy) > 0:
                b = pose_boxes.xyxy[0].to('cpu').detach().numpy().copy()
                px1, py1, px2, py2 = map(int, b[:4])
                # print(px1, py1, px2, py2)
                # cv2.putText(pose_color_image, "left", (px1, py1+50), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.7, (0, 255, 0), 2, cv2.LINE_4)
                # cv2.putText(pose_color_image, "right", (px2, py2), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.7, (0, 255, 0), 2, cv2.LINE_4)
            for i, k in enumerate(keypoints):
                if k.xy[0].size(0) > 0:  # Ensure there are enough elements
                    coordinate_pose[6] = k.xy[0][0].cpu().numpy() #Nose
                    depth_nose = depth_frame.get_distance(int(coordinate_pose[6][0]), int(coordinate_pose[6][1]))
                    cv2.putText(pose_color_image, f"Depth: {depth_nose}", (int(coordinate_pose[6][0]),int(coordinate_pose[6][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_4)   
                if k.xy[0].size(0) > 6:  
                    coordinate_pose[0] = k.xy[0][6].cpu().numpy()
                    value_srx = int(coordinate_pose[0][0]) # Right shoulder
                    
                if k.xy[0].size(0) > 8:  
                    coordinate_pose[1] = k.xy[0][8].cpu().numpy()  # Right elbow
                    
                if k.xy[0].size(0) > 10:  
                    coordinate_pose[2] = k.xy[0][10].cpu().numpy() # Right wrist
                    
                    if box_cx is not None:
                        distance_whr = np.sqrt((box_cx - int(coordinate_pose[2][0]))**2 + (box_cy - int(
                            coordinate_pose[2][1]))**2) 
                        
                if k.xy[0].size(0) > 5:
                    coordinate_pose[3] = k.xy[0][5].cpu().numpy()
                    value_slx = int(coordinate_pose[3][0]) # Left shoulder

                if k.xy[0].size(0) > 7:
                    coordinate_pose[4] = k.xy[0][7].cpu().numpy() # Left elbow

                if k.xy[0].size(0) > 9:
                    coordinate_pose[5] = k.xy[0][9].cpu().numpy() # Left wrist

                    if box_cx is not None:
                        distance_whl = np.sqrt((box_cx - int(coordinate_pose[5][0]))**2 + (
                            box_cy - int(coordinate_pose[5][1]))**2) 
                
                
                if distance_whl is not None and distance_whr is not None: #  Activate hand selection
                    if (distance_whl > distance_whr):
                        active_hands = 'RIGHT'
                        angle_arm = calculate_angle_arm(
                            coordinate_pose[0], coordinate_pose[1], coordinate_pose[2])
                    elif (distance_whl < distance_whr):
                        active_hands = 'LEFT'
                        angle_arm = calculate_angle_arm(
                            coordinate_pose[3], coordinate_pose[4], coordinate_pose[5])
                        
                if box_cx is not None and box_cy is not None: #Figure 1: misrecognition bacground
                    if(box_cy<y1 or box_cy>y2 or box_cx<x1 or box_cx>x2):
                        print("out of box\n")
                        gesture='N'
                        
                box_cx, box_cy = None, None
                
    gesture_this = final_hands
    if gesture_this =='P' and pbox_cx is not None:
        if active_hands == 'RIGHT'and distance_whr is not None and value_srx is not None: 
            if pbox_cx > value_srx:
                gesture = 'R'
            else:
                gesture = 'L'
        elif active_hands == 'LEFT'and distance_whl is not None and value_slx is not None:
            if pbox_cx > value_slx:
                gesture = 'R'
            else:
                gesture = 'L'
    elif gesture_this =='W':
        gesture='W'
    elif(gesture_this==gesture_pre):
        count_gesture+=1
        if(count_gesture>3):
            count_gesture=0
            gesture=gesture_this
            # print('gesture: ',gesture)
    else:
        gesture_pre  = gesture_this

    if gesture != 'N':
        # print(gesture, angle_arm)
        print(gesture)
        # ser.write(str(gesture).encode('utf-8')) #To do
        gesture = 'N'
        # count_print = 0
        # time2 = time.time()
        # print(f"FPS : {1 / (time2 - time1):.2f}")

    # cv2.imshow("predict", color_image)  # 주석 처리된 부분은 필요에 따라 활성화할 수 있습니다.
    cv2.imshow("predict", pose_color_image)  # 주석 처리된 부분은 필요에 따라 활성화할 수 있습니다.

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        ser.close()

pipeline.stop()  # 카메라 파이프라인을 종료합니다.
