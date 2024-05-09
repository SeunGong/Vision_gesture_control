#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pickle import NONE
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


# Road YOLO model
model_pose = YOLO("yolov8m-pose")
model_hands = YOLO("240503.pt")

box_cx, box_cy = None, None  # predict box
pbox_cx, pbox_cy = None, None  # pointing box

# Previous center coordinates
pre_stop_cx, pre_stop_cy = None, None
cur_stop_cx, cur_stop_cy = None, None
threshold_waving = 40  # Threshold for waving

angle_arm = 0
count_gesture = 0
pre_gesture = 'N'
shape_hand = 'N'
depth_hand = None
count_keypoints = 9  # Number of array for pose coordinate
active_hands = None

while True:
    # time1 = time.time() #for measure FPS
    # Searching color and depth image from RealSense
    frames = pipeline.wait_for_frames()
    # Aling depth frame from color frame aspect
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if not color_frame:  # Skip frame dosen't exigst color image data
        continue

    # Transform frame data to Numpy array
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    results_hands = model_hands(
        color_image, conf=0.8, verbose=False)  # Predict hands

    if results_hands is not None:

        # pre_stop_cx = None
        # pre_stop_cy = None

        for r in results_hands:
            boxes = r.boxes  # Boxes class

            array_depth_boxes = [0] * len(boxes)
            array_shape_hands = [''] * len(boxes)
            min_box_depth = float('inf')
            final_hands_index = 0

            for number_box, box in enumerate(boxes):  # Multi box detect
                b = box.xyxy[0].to('cpu').detach().numpy().copy()
                # Box left top and right bottom coordinate
                x1, y1, x2, y2 = map(int, b[:4])
                box_cx, box_cy = int(
                    (x2 - x1) / 2 + x1), int((y2 - y1) / 2 + y1)  # Get box center coordinate
                depth_box = depth_frame.get_distance(box_cx, box_cy)
                array_depth_boxes[number_box] = depth_box
                array_shape_hands[number_box] = model_hands.names[int(box.cls)]

                # Drawing bounding box
                if depth_box < min_box_depth:
                    min_box_depth = depth_box
                    final_hands_index = number_box

                cv2.rectangle(color_image, (x1, y1), (x2, y2),
                              (0, 0, 255), thickness=2, lineType=cv2.LINE_4)
                cv2.putText(color_image, f"Depth: {depth_box}", (
                    x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(color_image, array_shape_hands[number_box], (
                    box_cx, box_cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if len(array_shape_hands) > final_hands_index:  # Ensure index is within bounds
                shape_hand = array_shape_hands[final_hands_index]
                depth_hand = array_depth_boxes[final_hands_index]
                # print("find min hands:",shape_hand)
                # print(f"Closest Hand Gesture: {shape_hand} at Index {final_hands_index} with Depth {min_box_depth}")

                if shape_hand == 'STOP':
                    cur_stop_cx, cur_stop_cy = int(
                        (x2 - x1) / 2 + x1), int((y2 - y1) / 2 + y1)  # Set current box coordinate
                    if pre_stop_cx is not None and pre_stop_cy is not None:
                        distance_stop = abs(
                            cur_stop_cx - pre_stop_cx)  # Get distance using x
                        # print(distance_stop)

                        if distance_stop > threshold_waving:  # Check sharply movement using only x value
                            shape_hand = 'W'
                        else:
                            shape_hand = 'S'

                    pre_stop_cx, pre_stop_cy = cur_stop_cx, cur_stop_cy

                elif shape_hand == 'POINTING':
                    shape_hand = shape_hand[0]
                    pbox_cx, pbox_cy = box_cx, box_cy
                else:
                    shape_hand = shape_hand[0]

    distance_whl, distance_whr = None, None  # Distance both side winkle-hands
    diff_x_sh = None  # Difference between shoulder and hip
    value_slx, value_srx = None, None  # Value shoulder x
    value_slx, value_srx = None, None  # Value shoulder x
    value_sly, value_sry = None, None  # Value shoulder y
    value_hly, value_hry = None, None  # Value hip right x
    depth_nose = None

    results_pose = model_pose(color_image, conf=0.8,
                              verbose=False)  # Predict pose
    pose_color_image = results_pose[0].plot()  # Draw skelton to pose image

    array_keypoints = np.zeros((count_keypoints, 2))  # [RS,RE,RW,LS,LE,LW,N]
    if results_pose is not None:
        for r in results_pose:
            keypoints = r.keypoints
            pose_boxes = r.boxes
            
            min_pose_depth = float('inf')
            final_pose_index = 0
            abc=0
            #Finding for 
            for number_pose_box, box in enumerate(pose_boxes):
                # abc+=1 #check for pose box count
                b = box.xyxy[0].to('cpu').detach().numpy().copy()
                # Box left top and right bottom coordinate
                x1, y1, x2, y2 = map(int, b[:4])
                pose_box_cx, pose_box_cy = int(
                    (x2 - x1) / 2 + x1), int((y2 - y1) / 2 + y1)  # Get box center coordinate
                depth_pose_box = depth_frame.get_distance(pose_box_cx, pose_box_cy)
                
                # Drawing bounding box
                if depth_pose_box < min_pose_depth:
                    # cv2.putText(pose_color_image, "left", (pose_box_cx, pose_box_cy+10), cv2.FONT_HERSHEY_SIMPLEX,
                    #     0.7, (0, 255, 0), 2, cv2.LINE_4)
                    min_pose_depth = depth_pose_box
                    final_pose_index = number_pose_box
                    # px1, py1, px2, py2= map(int, b[:4])
                    
            coordi_pose=pose_boxes[final_pose_index].xyxy[0]
            px1, py1, px2, py2= map(int, coordi_pose[:4])      
            cv2.putText(pose_color_image, "left", (px1, py1+10), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2, cv2.LINE_4)
            cv2.putText(pose_color_image, "right", (px2, py2), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2, cv2.LINE_4) 
            
            # print(abc)       
            # abc=0
            
            # if len(pose_boxes.xyxy) > 0:
            #     # print(px1, py1, px2, py2)

            for i, k in enumerate(keypoints[final_pose_index]):
                if k.xy[0].size(0) > 0:  # Ensure there are enough elements
                    array_keypoints[6] = k.xy[0][0].cpu().numpy()  # Nose
                    try:
                        depth_nose = depth_frame.get_distance(
                            int(array_keypoints[6][0]), int(array_keypoints[6][1]))
                    except RuntimeError as e:
                        print(f"An error occurred: {e}")

                    cv2.putText(pose_color_image, f"Depth: {depth_nose}", (int(array_keypoints[6][0]), int(
                        array_keypoints[6][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_4)

                    if (depth_nose is not None and depth_hand is not None):
                        if (depth_nose != 0.0 and depth_hand != 0.0):
                            if (depth_nose-depth_hand < 0):
                                continue
                            # print('{0:.2f} {1} {2}'.format(depth_nose-depth_hand, depth_nose, depth_hand))

                if k.xy[0].size(0) > 6:
                    array_keypoints[0] = k.xy[0][6].cpu().numpy()
                    value_srx = int(array_keypoints[0][0])  # Right shoulder x
                    value_sry = int(array_keypoints[0][1])  # Right Shoulder y
                if k.xy[0].size(0) > 8:
                    # Right elbow
                    array_keypoints[1] = k.xy[0][8].cpu().numpy()

                if k.xy[0].size(0) > 10:
                    # Right wrist
                    array_keypoints[2] = k.xy[0][10].cpu().numpy()

                    if box_cx is not None:
                        distance_whr = np.sqrt((box_cx - int(array_keypoints[2][0]))**2 + (box_cy - int(
                            array_keypoints[2][1]))**2)
                if k.xy[0].size(0) > 12:
                    # Right hip
                    array_keypoints[7] = k.xy[0][12].cpu().numpy()
                    value_hry = int(array_keypoints[7][1])
                if k.xy[0].size(0) > 5:
                    array_keypoints[3] = k.xy[0][5].cpu().numpy()
                    value_slx = int(array_keypoints[3][0])  # Left shoulder
                    value_sly = int(array_keypoints[3][1])  # Left shoulder

                if k.xy[0].size(0) > 7:
                    array_keypoints[4] = k.xy[0][7].cpu().numpy()  # Left elbow

                if k.xy[0].size(0) > 9:
                    array_keypoints[5] = k.xy[0][9].cpu().numpy()  # Left wrist

                    if box_cx is not None:
                        distance_whl = np.sqrt((box_cx - int(array_keypoints[5][0]))**2 + (
                            box_cy - int(array_keypoints[5][1]))**2)
                if k.xy[0].size(0) > 11:
                    # left hip
                    array_keypoints[8] = k.xy[0][11].cpu().numpy()
                    value_hly = int(array_keypoints[8][1])

                if distance_whl is not None and distance_whr is not None:  # Activate hand selection
                    if (distance_whl > distance_whr):
                        active_hands = 'RIGHT'
                        angle_arm = calculate_angle_arm(
                            array_keypoints[0], array_keypoints[1], array_keypoints[2])
                    elif (distance_whl < distance_whr):
                        active_hands = 'LEFT'
                        angle_arm = calculate_angle_arm(
                            array_keypoints[3], array_keypoints[4], array_keypoints[5])

                if box_cx is not None and box_cy is not None:  # Figure 1: misrecognition background
                    if (box_cy < y1 or box_cy > y2 or box_cx < x1 or box_cx > x2):
                        print("Misrecognition gesture out of the box.\n")
                        gesture = 'N'
                    # Get ratio between shoulder-hip and shoulder-box
                    if (active_hands == 'RIGHT' and value_hry is not None and value_sry is not None):
                        if(value_hry>0 and value_sry>0):
                            diff_y_sh = value_hry-value_sry
                            diff_y_sb = abs(box_cy-value_sry)
                            print(diff_y_sb/diff_y_sh, 'H: ', value_hry,
                                'S: ', value_sry, 'B: ', box_cy)
                    elif (active_hands == 'LEFT' and value_hly is not None and value_sly is not None):
                        if(value_hly>0 and value_sly>0):
                            diff_y_sh = value_hly-value_sly
                            diff_y_sb = abs(box_cy-value_sly)
                            print(diff_y_sb/diff_y_sh, 'H: ', value_hly,
                                'S: ', value_sly, 'B: ', box_cy)

                box_cx, box_cy = None, None

    gesture_this = shape_hand
    if gesture_this == 'P' and pbox_cx is not None:
        if active_hands == 'RIGHT' and distance_whr is not None and value_srx is not None:
            if pbox_cx > value_srx:
                gesture = 'R'
            else:
                gesture = 'L'
        elif active_hands == 'LEFT' and distance_whl is not None and value_slx is not None:
            if pbox_cx > value_slx:
                gesture = 'R'
            else:
                gesture = 'L'
    elif gesture_this == 'W':
        gesture = 'W'
    elif (gesture_this == pre_gesture):
        count_gesture += 1
        if (count_gesture > 3):
            count_gesture = 0
            gesture = gesture_this
            # print('gesture: ',gesture)
    else:
        pre_gesture = gesture_this

    if gesture != 'N':
        # print(gesture, angle_arm)
        # print(gesture)
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
