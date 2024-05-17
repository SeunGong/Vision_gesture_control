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
from predict_f import calculate_angle_arm

# Serial setting
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

# final_hand = None
count_print = 0

# Road YOLO model
model_pose = YOLO("yolov8m-pose")
model_hands = YOLO("240503.pt")

# MACRO
THRESHOLD_WAVING = 100  # Threshold for waving
COUNT_GESTURE = 0
WEIGHT_DIRECTION = 0.0045
WEIGHT_DEPTH = 0.9
DEPTH_DISTANCE_MAX = 4
MOTOR_ENCODER = 4096
MOTOR_DISTANCE = 0.534
# Init variable
pre_stop_cx, pre_stop_cy = 0, 0  # Previous center coordinates
pre_gesture = "N"
# Flag
flag_init_stop_x = True

keypoints_count = 9  # Number of array for pose coordinate
keypoint_indices = {
    0: 6,  # Nose
    6: 0,  # Right shoulder
    8: 1,  # Right elbow
    10: 2,  # Right wrist
    12: 7,  # Right hip
    5: 3,  # Left shoulder
    7: 4,  # Left elbow
    9: 5,  # Left wrist
    11: 8,  # Left hip
}

while True:
    # Get camera frame#########################################
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not color_frame:  # Skip frame dosen't exigst color image data
        continue
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # Reset variable
    box_cx, box_cy = None, None  # hands box center
    cur_cx, cur_cy = None, None
    # pbox_cx, pbox_cy = None, None  # pointing box center
    lsx, lsy, rsx, rsy = None, None, None, None  # shoulder x,y
    lhy, rhy = None, None  # hip y
    euclidean_whl, euclidean_whr = None, None  # Distance both side winkle-hands
    sb_sub, sh_sub = None, None

    depth_nose = None
    depth_hand = None
    pose_depth = None

    arm_angle = None
    arm_ratio = None

    active_hand = None
    shape_hand = None
    ratio_hand = "N"
    final_hand = "N"

    box_depth = None

    motor_L, motor_R = 0, 0
    array_keypoints = np.zeros((keypoints_count, 2))  # [RS,RE,RW,LS,LE,LW,]

    #################### Predict hands ####################
    results_hands = model_hands(color_image, conf=0.8, verbose=False)
    if results_hands is not None:
        for r in results_hands:
            boxes = r.boxes  # Boxes class
            final_hands_index = 0
            box_depth = float("inf")
            list_depth_boxes = [0] * len(boxes)
            list_shape_hands = [""] * len(boxes)
            x1, y1, x2, y2 = 0, 0, 0, 0

            # Check front hand
            for number_box, box in enumerate(boxes):
                b = box.xyxy[0].to("cpu").detach().numpy().copy()
                x1, y1, x2, y2 = map(
                    int, b[:4]
                )  # Box left top and right bottom coordinate
                box_cx, box_cy = int((x2 - x1) / 2 + x1), int((y2 - y1) / 2 + y1)
                depth_box = depth_frame.get_distance(box_cx, box_cy)
                list_depth_boxes[number_box] = depth_box
                list_shape_hands[number_box] = model_hands.names[int(box.cls)]

                # Drawing bounding box
                if depth_box != 0 and depth_box < box_depth:
                    box_depth = depth_box
                    final_hands_index = number_box

            # select active hand
            if len(boxes) > 0:  # Ensure index is within bounds
                x1, y1, x2, y2 = map(
                    int,
                    boxes[final_hands_index].xyxy[0].to("cpu").detach().numpy().copy(),
                )
                box_cx, box_cy = int((x2 - x1) / 2 + x1), int((y2 - y1) / 2 + y1)
                shape_hand = list_shape_hands[final_hands_index]
                depth_hand = list_depth_boxes[final_hands_index]

                cv2.rectangle(
                    color_image,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    thickness=2,
                    lineType=cv2.LINE_4,
                )
                # cv2.putText(color_image, f"Depth: {depth_hand}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(
                    color_image,
                    shape_hand,
                    (box_cx, box_cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                # print("find min hands:",shape_hand)
                # print(f"Closest Hand Gesture: {shape_hand} at Index {final_hands_index} with Depth {box_depth}")
                # convert 1 character

                # Get hand shape
                if shape_hand == "STOP":
                    shape_hand = shape_hand[0]
                    cur_cx, cur_cy = int((x2 - x1) / 2 + x1), int(
                        (y2 - y1) / 2 + y1
                    )  # Set current box coordinate
                    if flag_init_stop_x == True:
                        flag_init_stop_x = False
                        pre_stop_cx = cur_cx

                    # if pre_stop_cx is not None and pre_stop_cy is not None:
                    #     distance_stop = abs(cur_cx - pre_stop_cx)  # Get distance using x
                    #     # print(distance_stop)
                    #     if distance_stop > THRESHOLD_WAVING:  # Check sharply movement using only x value
                    #         shape_hand = 'W'

                    # pre_stop_cx, pre_stop_cy = cur_cx, cur_cy
                # elif shape_hand == 'POINTING':
                #     shape_hand = shape_hand[0]
                #     # pbox_cx, pbox_cy = box_cx, box_cy
                else:
                    shape_hand = shape_hand[0]

    #################### Predict pose ####################
    results_pose = model_pose(color_image, conf=0.8, verbose=False)  # Predict pose
    pose_color_image = results_pose[0].plot()  # Draw skelton to pose image

    if results_pose is not None:
        for r in results_pose:
            keypoints = r.keypoints
            pose_boxes = r.boxes

            pose_depth = float("inf")
            final_pose_index = 0

            # Check front pose
            for number_pose_box, box in enumerate(pose_boxes):
                b = box.xyxy[0].to("cpu").detach().numpy().copy()
                x1, y1, x2, y2 = map(int, b[:4])
                pose_box_cx, pose_box_cy = int((x2 - x1) / 2 + x1), int(
                    (y2 - y1) / 2 + y1
                )  # Get box center coordinate
                depth_pose_box = depth_frame.get_distance(pose_box_cx, pose_box_cy)

                if depth_pose_box < pose_depth:
                    pose_depth = depth_pose_box
                    final_pose_index = number_pose_box
            # Check index count more than pose count
            if len(pose_boxes) > final_pose_index:
                coordi_pose = pose_boxes[final_pose_index].xyxy[0]
                px1, py1, px2, py2 = map(int, coordi_pose[:4])
                # cv2.putText(pose_color_image, "left", (px1, py1+10), cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7, (0, 255, 0), 2, cv2.LINE_4)
                # cv2.putText(pose_color_image, "right", (px2, py2), cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7, (0, 255, 0), 2, cv2.LINE_4)
            # Get keypoints
            for i, k in enumerate(keypoints[final_pose_index]):
                for kp_index, ap_index in keypoint_indices.items():
                    if k.xy[0].size(0) > kp_index:
                        array_keypoints[ap_index] = k.xy[0][kp_index].cpu().numpy()
            # put value in variable
            lsx = int(array_keypoints[3][0])
            lsy = int(array_keypoints[3][1])
            rsx = int(array_keypoints[0][0])
            rsy = int(array_keypoints[0][1])
            rhy = int(array_keypoints[7][1])
            lhy = int(array_keypoints[8][1])
    ###################################################################################

    # Distinction between left and right hands
    if box_cx is not None and box_cy is not None:
        euclidean_whr = np.sqrt(
            (box_cx - int(array_keypoints[2][0])) ** 2
            + (box_cy - int(array_keypoints[2][1])) ** 2
        )
        euclidean_whl = np.sqrt(
            (box_cx - int(array_keypoints[5][0])) ** 2
            + (box_cy - int(array_keypoints[5][1])) ** 2
        )

        # Activate hand selection
        if euclidean_whl is not None and euclidean_whr is not None:
            if euclidean_whl > euclidean_whr:
                active_hand = "RIGHT"
                arm_angle = calculate_angle_arm(
                    array_keypoints[0], array_keypoints[1], array_keypoints[2]
                )
            elif euclidean_whl < euclidean_whr:
                active_hand = "LEFT"
                arm_angle = calculate_angle_arm(
                    array_keypoints[3], array_keypoints[4], array_keypoints[5]
                )

            # Get ratio between shoulder-hip and shoulder-box
            if active_hand == "RIGHT" and rhy is not None and rsy is not None:
                if rhy > 0 and rsy > 0:
                    sh_sub = rhy - rsy
                    sb_sub = abs(box_cy - rsy)
                    arm_ratio = sb_sub / sh_sub
            elif active_hand == "LEFT" and lhy is not None and lsy is not None:
                if lhy > 0 and lsy > 0:
                    sh_sub = lhy - lsy
                    sb_sub = abs(box_cy - lsy)
                    arm_ratio = sb_sub / sh_sub

            # Check arm_ratio and arm_angle
            if shape_hand == "S":
                # if(shape_hand=='S' or shape_hand=='W'):

                if pre_stop_cx is not None and lsx is not None and rsx is not None:
                    THRESHOLD_WAVING = lsx - rsx
                    distance_stop = abs(cur_cx - pre_stop_cx)  # Get distance using x
                    if (
                        distance_stop > THRESHOLD_WAVING
                    ):  # Check sharply movement using only x value
                        shape_hand = "W"
                    # print(cur_cx,pre_stop_cx,flag_init_stop_x)
                    # print(distance_stop,THRESHOLD_WAVING)

                    pre_stop_cx = cur_cx

                ratio_hand = shape_hand
            elif arm_ratio is not None and arm_ratio < 0.3:
                if shape_hand == "T":
                    ratio_hand = shape_hand
                elif shape_hand == "Y":
                    ratio_hand = shape_hand
            elif arm_ratio is not None and arm_ratio > 0.45:
                if shape_hand == "F":
                    ratio_hand = shape_hand
                elif shape_hand == "B" and arm_angle < 120:
                    ratio_hand = shape_hand
                elif shape_hand == "P":
                    if box_depth > DEPTH_DISTANCE_MAX:
                        box_depth = DEPTH_DISTANCE_MAX
                    elif box_depth < 0:
                        box_depth = 0
                    distance_depth = box_depth * WEIGHT_DEPTH
                    if box_cx > 320:
                        box_center_sub = box_cx - 320
                        # move_direction = 'R'
                        motor_L = distance_depth + (box_center_sub * WEIGHT_DIRECTION)
                        motor_R = distance_depth
                    elif box_cx < 320:
                        box_center_sub = 320 - box_cx
                        # move_direction = 'L'
                        motor_L = distance_depth
                        motor_R = distance_depth + (box_center_sub * WEIGHT_DIRECTION)
                    else:
                        motor_L = distance_depth
                        motor_R = distance_depth
                    motor_L = format((int)(motor_L * 100), "03")
                    motor_R = format((int)(motor_R * 100), "03")
                    # motor_encoder_L=MOTOR_ENCODER*motor_L/MOTOR_DISTANCE
                    # motor_encoder_R=MOTOR_ENCODER*motor_R/MOTOR_DISTANCE
                    ratio_hand = shape_hand
                    # print(f"L:{motor_L:.2f}, R:{motor_R:.2f}, DEPTH:{box_depth:.2f},E_L:{motor_encoder_L},E_R:{motor_encoder_R}")

                    # if active_hand == 'RIGHT' and euclidean_whr is not None and rsx is not None:
                    #     if box_cx > rsx:
                    #         ratio_hand = 'R'
                    #     else:
                    #         ratio_hand = 'L'
                    # elif active_hand == 'LEFT' and euclidean_whl is not None and lsx is not None:
                    #     if box_cx > lsx:
                    #         ratio_hand = 'R'
                    #     else:
                    #         ratio_hand = 'L'

            # if(arm_ratio is not None and arm_angle is not None):
            #     print(f"Hand: {ratio_hand}",f"Ratio: {arm_ratio:.3f}",f"angle: {arm_angle:.3f}")

        # #Check out of boundary box
        # if (box_cy < py1 or box_cy > py2 or box_cx < px1 or box_cx > px2):
        #     print("Misrecognition hands out of the box.\n")
        #     final_hand = 'N'

        # 3times validation
        this_hand = ratio_hand
        # if (this_hand == 'W'or this_hand =='P'):
        if this_hand == "W":
            final_hand = this_hand
        elif this_hand == pre_gesture:
            COUNT_GESTURE += 1
            if COUNT_GESTURE > 3 and this_hand != "T":
                COUNT_GESTURE = 0
                final_hand = this_hand
            elif COUNT_GESTURE > 5:
                COUNT_GESTURE = 0
                final_hand = this_hand

        if final_hand == "P":
            print(f"<L{motor_L}R{motor_R}>")
            # ser.write(str(f"<{final_hand}>")).encode('utf-8'))
            ser.write(str(f"<L{motor_L}R{motor_R}>").encode("utf-8"))
        elif final_hand != "N" and final_hand != "P":
            print(f"<{final_hand}0000000>")

            ser.write(str(f"<{final_hand}0000000>").encode("utf-8"))
            final_hand = "N"
        else:
            pre_gesture = this_hand

    # cv2.imshow("predict", pose_color_image)  # 주석 처리된 부분은 필요에 따라 활성화할 수 있습니다.

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        ser.close()

pipeline.stop()  # 카메라 파이프라인을 종료합니다.
