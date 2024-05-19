#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pickle import NONE
import cv2
import time
import serial
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import platform

from ultralytics import YOLO
from predict_f import *

from collections import deque

# Initialize variables
pre_stop_positions = deque(maxlen=5)  # Keep the last 5 positions
pre_stop_positions.append((0, 0))  # Initial position

# Serial setting
if platform.system() == "Linux":
    ser = serial.Serial("/dev/ttyUSB0", 115200)

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


# Road YOLO model
model_pose = YOLO("yolov8m-pose")
model_hands = YOLO("240503.pt")

# MACRO
WEIGHT_DIRECTION = 0.0045
WEIGHT_DEPTH = 0.9
DEPTH_DISTANCE_MAX = 4
MOTOR_ENCODER = 4096
MOTOR_DISTANCE = 0.534

count_gesture = 0
count_turn_gesture = 0

# Init variable
pre_stop_cx, pre_stop_cy = 0, 0  # Previous center coordinates
pre_gesture = "N"

# Flag
flag_init_stop_x = True
waving_flag = False

keypoints_count = 9  # Number of array for pose coordinate
keypoint_indices = {
    6: 0,  # Right shoulder
    8: 1,  # Right elbow
    10: 2,  # Right wrist
    5: 3,  # Left shoulder
    7: 4,  # Left elbow
    9: 5,  # Left wristD
    0: 6,  # Nose
    12: 7,  # Right hip
    11: 8,  # Left hip
}

array_keypoints = np.zeros((keypoints_count, 2))  # [RS,RE,RW,LS,LE,LW,]
print ("start!!!")


while True:
    # Get camera frame#########################################
    time1 = time.time()
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not color_frame:  # Skip frame dosen't exist color image data
        continue
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    flag_continue = False

    #################### Predict active hands ####################
    results_hands = model_hands(color_image, conf=0.8, verbose=False)
    if results_hands is not None:
        for r in results_hands:
            boxes = r.boxes  # Boxes class
            active_depth_box = float("inf")


            # Check front hand
            for number_box, box in enumerate(boxes):
                x1, y1, x2, y2, box_cx, box_cy, depth_hand_box, shape_hand = get_box_coordinates(
                    box, depth_frame, model_hands, DEPTH_DISTANCE_MAX)

                # Drawing bounding box
                if depth_hand_box != 0 and depth_hand_box < active_depth_box:
                    active_depth_hand = depth_hand_box
                    active_box_cx = box_cx
                    active_box_cy = box_cy
                    active_x1, active_y1, active_x2, active_y2 = x1, y1, x2, y2
                    active_shape_hand = shape_hand
                else:
                    flag_continue = True

            # select active hand
            if len(boxes) > 0:  # Ensure index is within bounds
                # cv2.rectangle(color_image, (active_x1, active_y1), (active_x2, active_y2), (0, 0, 255), thickness=2, lineType=cv2.LINE_4,)
                # cv2.putText(color_image, f"Depth: {active_depth_hand}", (active_x1, active_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(
                    color_image,
                    active_shape_hand,
                    (active_box_cx, active_box_cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

                # Get hand shape
                cur_stop_cx, cur_stop_cy = active_box_cx, active_box_cy
                shape_hand = active_shape_hand[0]

            else:
                flag_continue = True
                continue
    else: 
        continue
    if flag_continue:
        continue


    #################### Predict pose ####################
    results_pose = model_pose(color_image, conf=0.8, verbose=False)  # Predict pose
    # pose_color_image = results_pose[0].plot()  # Draw skelton to pose image
    # cv2.imshow("predict", pose_color_image)
    cv2.imshow("predict", color_image) 
    if results_pose is not None:
        for r in results_pose:
            keypoints = r.keypoints
            pose_boxes = r.boxes
            active_depth_pose = float("inf")

            # Check front pose
            for number_pose_box, box in enumerate(pose_boxes):
                b = box.xyxy[0].to("cpu").detach().numpy().copy()
                px1, py1, px2, py2 = map(int, b[:4])
                pose_box_cx, pose_box_cy = int((px2 - px1) / 2 + px1), int(
                    (py2 - py1) / 2 + py1
                )  # Get box center coordinate
                depth_pose_box = depth_frame.get_distance(pose_box_cx, pose_box_cy)

                if depth_pose_box < active_depth_pose:
                    active_depth_pose = depth_pose_box
                    active_px1, active_py1, active_px2, active_py2 = px1, py1, px2, py2                   
                    
                    # Get keypoints
                    for i, k in enumerate(keypoints[number_pose_box]):
                        for kp_index, ap_index in keypoint_indices.items():
                            if k.xy[0].size(0) > kp_index:
                                array_keypoints[ap_index] = k.xy[0][kp_index].cpu().numpy()                    

            # Check index count more than pose count
            if len(pose_boxes) > 0:
                lsx, lsy = map(int, array_keypoints[3]) # Left shoulder
                rsx, rsy = map(int, array_keypoints[0]) # Right shoulder
                lhy = int(array_keypoints[8][1])        # Left hip
                rhy = int(array_keypoints[7][1])        # Right hip
                lwx, lwy = map(int, array_keypoints[5]) # Left wrist
                rwx, rwy = map(int, array_keypoints[2]) # Right wrist
                coordinates = [lsx, lsy, rsx, rsy, lhy, rhy, lwx, lwy, rwx, rwy]
                # cv2.imshow("predict", pose_color_image)

                if any(coord == 0 for coord in coordinates):
                    flag_continue = True
                    continue
            else:
                flag_continue = True
                continue
    else: 
        continue
    if flag_continue:
        continue


    ###############################Post-processing###################################
    active_hand, arm_angle, arm_ratio = select_active_hand(box_cx, box_cy, array_keypoints)
    if arm_ratio is None:
        continue

    # Check arm_ratio and arm_angle
    ratio_hand = "N"
    if shape_hand == "S":
        ratio_hand = shape_hand
        
        if flag_init_stop_x == True:
            flag_init_stop_x = False
            pre_stop_positions[-1] = (cur_stop_cx, cur_stop_cy)
            continue

        threshold_waving_y = 70
        pre_stop_positions.append((cur_stop_cx, cur_stop_cy))

        if len(pre_stop_positions) == 5:
            x_positions = [x for x, y in pre_stop_positions]
            y_positions = [y for x, y in pre_stop_positions]

            # Check sharply movement
            x_variance = max(x_positions) - min(x_positions)
            y_variance = max(y_positions) - min(y_positions)
            if x_variance > (lsx - rsx) * 0.9 and y_variance < threshold_waving_y:
                ratio_hand = "W"


    elif shape_hand == "T":
        if(arm_ratio < 0.3):
            ratio_hand = shape_hand
    elif shape_hand == "Y":
        if(arm_ratio < 0.3):                    
            ratio_hand = shape_hand
    elif shape_hand == "F":
        if(arm_ratio > 0.45 and arm_ratio<0.8):
            ratio_hand = shape_hand
    elif shape_hand == "B": 
        # if(arm_ratio > 0.45 and arm_angle < 120):
        if(arm_ratio > 0.45):
            ratio_hand = shape_hand
    elif shape_hand == "P":
        if(arm_ratio > 0.45):
            if active_box_cx >= (3 * lsx + rsx) / 4:
                ratio_hand = "L"
            elif (3 * lsx + rsx) / 4 > active_box_cx and active_box_cx >= (lsx + 3 * rsx) / 4:
                ratio_hand = "F"
            else:
                ratio_hand = "R"

    # 3 times in-a-row validation
    this_hand = ratio_hand
    final_hand = "N"

    if this_hand == "T":
        count_turn_gesture += 1
        count_gesture = 0
        if count_turn_gesture >= 7:
            count_turn_gesture = 0
            final_hand = "T"

    elif this_hand == pre_gesture:
        count_gesture += 1
        if count_gesture > 2:
            final_hand = this_hand
            count_gesture -= 1
            count_turn_gesture -= 1
            if this_hand == "W":
                waving_flag = True
    else:
        count_gesture = 0
    
    if final_hand != "N":
        # print(f"<{final_hand}0000000>")
        # print(f"<{final_hand}0000000>,angle:{arm_angle:.2f},ratio:{arm_ratio:.2f}")
        if platform.system() == "Linux":
            ser.write(str(f"<{final_hand}000000>").encode("utf-8"))
    else:
        pre_gesture = this_hand
    

    # cv2.imshow("predict", pose_color_image)  # 주석 처리된 부분은 필요에 따라 활성화할 수 있습니다.
    print(f"<{shape_hand}{ratio_hand}{final_hand}>,angle:{arm_angle:.2f},ratio:{arm_ratio:.2f}")
    if waving_flag:
        time.sleep(2)
        waving_flag = False

    time2 = time.time()
    # print("running time: ", time2 - time1)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        if platform.system() == "Linux":
            ser.close()
        break

pipeline.stop()  # 카메라 파이프라인을 종료합니다.