#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import time
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 카메라 프레임의 원하는 너비와 높이를 정의합니다.
W, H = 640, 480

# RealSense 카메라 파이프라인 초기화
config = rs.config()
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)

pipeline = rs.pipeline()
profile = pipeline.start(config)

# 컬러와 깊이 이미지 스트림의 정렬을 설정합니다.
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

# YOLOv8 모델을 로드합니다.
model_pose = YOLO("yolov8m-pose")
model_hands = YOLO("240503.pt")

box_cx, box_cy = None, None  # predict box
pbox_cx, pbox_cy = None, None  # pointing box

# Previous center coordinates
pre_stop_cx, pre_stop_cy, pre_cx_pointing, pre_cy_pointing = None, None, None, None
cur_stop_cx, cur_stop_cy, cur_cx_pointing, cur_cy_pointing = None, None, None, None
threshold_waving = 30  # Threshold for detecting significant change

angle_arm = 0

count_gesture=0
gesture_pre='N'

while True:
    # time1 = time.time() #for measure FPS
    frames = pipeline.wait_for_frames()  # RealSense로부터 컬러 및 깊이 이미지 프레임을 검색합니다.
    aligned_frames = align.process(frames)  # 깊이 프레임을 컬러 프레임의 관점으로 정렬합니다.
    
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if not color_frame:
        continue  # 컬러 이미지 데이터가 없으면 프레임을 건너뜁니다.

    # 프레임 데이터를 NumPy 배열로 변환합니다.
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    
    results_hands = model_hands(
        color_image, conf=0.8, verbose=False)  # Predict hands
    
    hands = 'N'
    
    if results_hands is not None:
        for r in results_hands:
            boxes = r.boxes
            for box in boxes:
                
                b = box.xyxy[0].to('cpu').detach().numpy().copy()
                c = box.cls
                x1, y1, x2, y2 = map(int, b[:4])
                box_cx, box_cy = int(
                    (x2 - x1) / 2 + x1), int((y2 - y1) / 2 + y1)
                array_boxes_depth = depth_frame.get_distance(box_cx, box_cy)
                # print(array_boxes_depth)
                hands = model_hands.names[int(c)]
                
                if hands == 'STOP':
                    cur_stop_cx, cur_stop_cy = int(
                        (x2 - x1) / 2 + x1), int((y2 - y1) / 2 + y1)
                    
                    hands = 'S'

                    if pre_stop_cx is not None and pre_stop_cy is not None:
                        # Calculate Euclidean distance between previous and current center
                        distance = np.sqrt(
                            (cur_stop_cx - pre_stop_cx)**2 + (cur_stop_cy - pre_stop_cy)**2)
                        # print(distance)
                        if distance > threshold_waving:
                            hands = 'W'
                        distance=0

                    pre_stop_cx, pre_stop_cy = cur_stop_cx, cur_stop_cy

                elif hands == 'FORWARD':
                    hands = 'F'
                elif hands == 'BACKWARD':
                    hands = 'B'
                elif hands == 'TURN':
                    hands = 'T'
                elif hands == 'YOU':
                    hands = 'Y'
                elif hands == 'POINTING':
                    hands = 'P'
                    pbox_cx, pbox_cy = box_cx, box_cy

                # Drawing bounding box
                cv2.rectangle(color_image, (x1, y1), (x2, y2),
                                  (0, 0, 255), thickness=2, lineType=cv2.LINE_4)
                cv2.putText(color_image,  model_hands.names[int(c)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2, cv2.LINE_4)


    results_pose = model_pose(color_image, conf=0.8, verbose=False) # Predict coordinate_pose

    pose_color_image = results_pose[0].plot()
    
    count_point = 6  # number of coordinate_pose
    
    distance_whl, distance_whr = None, None # Distance between winkle-hands
    value_slx, value_srx = None, None # Value shoulder x

    coordinate_pose = np.zeros((count_point, 2))
    if results_pose is not None:
        for r in results_pose:
            keypoints = r.keypoints
            pose_boxes = r.boxes
            if len(pose_boxes.xyxy) > 0:
                b = pose_boxes.xyxy[0].to('cpu').detach().numpy().copy()
                px1, py1, px2, py2 = map(int, b[:4])
                # print(px1, py1, px2, py2)
                cv2.putText(pose_color_image, "left", (px1, py1+50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2, cv2.LINE_4)
                cv2.putText(pose_color_image, "right", (px2, py2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2, cv2.LINE_4)
            for i, k in enumerate(keypoints):
                if k.xy[0].size(0) > 6:  # Ensure there are enough elements
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
                if box_cx is not None and box_cy is not None:
                    if(box_cy<y1 or box_cy>y2 or box_cx<x1 or box_cx>x2):
                        print("out of box\n")
                        # gesture=6
                        
                box_cx, box_cy = None, None
    # conditions = {
    #     "S": lambda angle_arm: angle_arm > 0 and angle_arm < 180,
    #     "F": lambda angle_arm: angle_arm > 80 and angle_arm < 180,
    #     "B": lambda angle_arm: angle_arm > 80 and angle_arm < 180,
    #     "T": lambda angle_arm: angle_arm > 0 and angle_arm < 60,
    #     "P": lambda angle_arm: angle_arm > 150 and angle_arm < 180,
    #     "Y": lambda angle_arm: angle_arm > 0 and angle_arm < 180,
    #     "W": lambda angle_arm: angle_arm > 0 and angle_arm < 180,
    # }

    # if conditions.get(hands, lambda x: False)(angle_arm):
    #     gesture_this = hands
    #     if gesture_this =='P' and pbox_cx is not None:
    #         if active_hands == 'RIGHT'and distance_whr is not None and value_srx is not None: 
    #             if pbox_cx > value_srx:
    #                 gesture = 'R'
    #             else:
    #                 gesture = 'L'
    #         elif active_hands == 'LEFT'and distance_whl is not None and value_slx is not None:
    #             if pbox_cx > value_slx:
    #                 gesture = 'R'
    #             else:
    #                 gesture = 'L'
    #     # print("this: ",gesture_this,"pre: ",gesture_pre,"count: ",count_gesture)

    #     if(gesture_this==gesture_pre):
    #         count_gesture+=1
    #         if(count_gesture>3):
    #             count_gesture=0
    #             gesture=gesture_this
    #             # print('gesture: ',gesture)
    #     else:
    #         gesture_pre  = gesture_this

    # else:
    #     gesture = 'N'
    gesture_this = hands
    
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
        gesture = 'N'
        # count_print = 0
        # time2 = time.time()
        # print(f"FPS : {1 / (time2 - time1):.2f}")

    cv2.imshow("predict", pose_color_image)  # 주석 처리된 부분은 필요에 따라 활성화할 수 있습니다.

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        ser.close()

pipeline.stop()  # 카메라 파이프라인을 종료합니다.
