import cv2
import serial
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from collections import deque, Counter

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

data_window_size=10
data_stream = [None] * data_window_size
data_number=0
data_final=None
window = deque(data_stream,maxlen=data_window_size)  # Fixed-size window
print_count=0

def update_list_and_find_mode(lst, new_value):
    global data_final
    if new_value is not None:
        lst.append(new_value)
        current_window_data = list(lst)
        mode_count = Counter(current_window_data).most_common(1)
        if mode_count:
                mode, count = mode_count[0]
                # print(f"Current window: {current_window_data} | Mode: {mode}, Count: {count}")
                data_final=mode
                
        else:
            print("No data available.")
                
    # for data_point in stream:  # Process each data point
    #     if data_point is not None:  # Only process if data_point is not None
    #         window.append(data_point)
    #         current_window_data = list(window)
    #         mode_count = Counter(current_window_data).most_common(1)
    #         if mode_count:
    #             mode, count = mode_count[0]
    #             # print(f"Current window: {current_window_data} | Mode: {mode}, Count: {count}")
    #             data_final=mode
    #         else:
    #             print("No data available.")


def calculate_angle(a, b, c):

    # 각 값을 받아 넘파이 배열로 변형
    a = np.array(a)  # 첫번째
    b = np.array(b)  # 두번째
    c = np.array(c)  # 세번째

    # 라디안을 계산하고 실제 각도로 변경한다.
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
                         np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    # 180도가 넘으면 360에서 뺀 값을 계산한다.
    if angle > 180.0:
        angle = 360-angle

    # 각도를 리턴한다.
    return angle


# YOLOv8 모델을 로드합니다.
model_pose = YOLO("yolov8m-pose")
model_hands = YOLO("240414.pt")

# Find hands.
object_name = 'N'
# Initialize variables outside of your main processing loop
# Previous center coordinates
prev_cx_stop, prev_cy_stop, prev_cx_move, prev_cy_move = None, None, None, None
current_cx_stop, current_cy_stop, current_cx_move, current_cy_move = None, None, None, None

box_cx,box_cy = None,None
pbox_cx,pbox_cy = None, None
change_threshold = 15  # Threshold for detecting significant change
angle = 0

while True:
    time1 = time.time()
    frames = pipeline.wait_for_frames()  # RealSense로부터 컬러 및 깊이 이미지 프레임을 검색합니다.

    aligned_frames = align.process(frames)  # 깊이 프레임을 컬러 프레임의 관점으로 정렬합니다.
    color_frame = aligned_frames.get_color_frame()
    if not color_frame:
        continue  # 컬러 이미지 데이터가 없으면 프레임을 건너뜁니다.

    # 프레임 데이터를 NumPy 배열로 변환합니다.
    color_image = np.asanyarray(color_frame.get_data())

    # YOLOv8 모델을 사용하여 컬러 이미지에서 탐지 결과를 얻습니다.
    results_hands = model_hands(color_image, conf=0.8, verbose=False)

    if results_hands is not None:
        object_name='N'
        for r in results_hands:
            boxes = r.boxes
            for box in boxes:
                # class_index = box.cls  # Get the class index of the object
                # Use the index to get the object's name
                b = box.xyxy[0].to('cpu').detach().numpy().copy()
                c = box.cls
                x1, y1, x2, y2 = map(int, b[:4])
                box_cx,box_cy=int((x2 - x1) / 2 + x1), int((y2 - y1) / 2 + y1) 
                
                object_name = model_hands.names[int(c)]
                
                if object_name == 'STOP':
                    current_cx_stop, current_cy_stop = int(
                        (x2 - x1) / 2 + x1), int((y2 - y1) / 2 + y1)
                    
                    if prev_cx_stop is not None and prev_cy_stop is not None:
                    # Calculate Euclidean distance between previous and current center
                        distance = np.sqrt(
                            (current_cx_stop - prev_cx_stop)**2 + (current_cy_stop - prev_cy_stop)**2)
                        # print(distance)

                        if distance > change_threshold:
                            object_name = 'W'
                        else:
                            object_name = 'S'

                    prev_cx_stop, prev_cy_stop = current_cx_stop, current_cy_stop
                
                else:
                    if object_name == 'FORWARD':
                        object_name='F'
                    elif object_name=='BACKWARD':
                        object_name='B'
                    elif object_name=='TURN':
                        object_name='T'
                    elif object_name == 'YOU':
                        object_name = 'Y'
                    elif object_name == 'POINTING':
                        object_name = 'P'
                        pbox_cx,pbox_cy=box_cx,box_cy
                        
                #Drawing bounding box      
                cv2.rectangle(color_image, (x1, y1), (x2, y2),
                (0, 0, 255), thickness=2, lineType=cv2.LINE_4)
                cv2.putText(color_image,  model_hands.names[int(c)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2, cv2.LINE_4)

    #Predict pose
    results_pose = model_pose(color_image, conf=0.8, verbose=False)

    pose_color_image = results_pose[0].plot()
    # Find pose.
    pointcount = 6 #number of skeleton_point
    distance_whl,distance_whr=None,None
    shoulder_L,shoulder_R=None,None
    hands=None
    
    skeleton_point = np.zeros((pointcount, 2))
    if results_pose is not None:
        for r in results_pose:
            keypoints = r.keypoints
            # print(keypoints.xy)  # 객체의 이름을 출력합니다.
            for i, k in enumerate(keypoints):
                if k.xy[0].size(0) > 6:  # Ensure there are enough elements
                    # Right shoulder
                    skeleton_point[0] = k.xy[0][6].cpu().numpy()
                    shoulder_R= int(skeleton_point[0][0])
                if k.xy[0].size(0) > 8:  # Ensure there are enough elements
                    skeleton_point[1] = k.xy[0][8].cpu().numpy()  # Right elbow
                if k.xy[0].size(0) > 10:  # Ensure there are enough elements
                    # Right wrist
                    skeleton_point[2] = k.xy[0][10].cpu().numpy()
                    
                    if box_cx is not None:
                        distance_whr = np.sqrt((box_cx - int(skeleton_point[2][0]))**2 + (box_cy - int(skeleton_point[2][1]))**2) #distance between right winkle and hands
                        # print(distance_whr)
                if k.xy[0].size(0) > 5: 
                    skeleton_point[3] = k.xy[0][5].cpu().numpy()
                    shoulder_L= int(skeleton_point[3][0])
                    
                if k.xy[0].size(0) > 7:  
                    skeleton_point[4] = k.xy[0][7].cpu().numpy()
                    
                if k.xy[0].size(0) > 9:  
                    skeleton_point[5] = k.xy[0][9].cpu().numpy()
                    
                    if box_cx is not None:
                        distance_whl = np.sqrt((box_cx - int(skeleton_point[5][0]))**2 + (box_cy - int(skeleton_point[5][1]))**2) #distance between left winkle and hands
                        # print(distance_whl)
                        box_cx,box_cy=None,None
                        
                #Distinguish which hands       
                if distance_whl is not None and distance_whr is not None: 
                    if(distance_whl>distance_whr):
                        hands='RIGHT'
                        angle = calculate_angle(
                        skeleton_point[0], skeleton_point[1], skeleton_point[2])
                    elif(distance_whl<distance_whr):
                        hands='LEFT'
                        angle = calculate_angle(
                        skeleton_point[3], skeleton_point[4], skeleton_point[5])
                    # print(hands)

    conditions = {
        # "F": lambda angle: angle > 130 and angle < 170,
        # "B": lambda angle: angle > 90 and angle < 130,
        "F": lambda angle: angle > 80 and angle < 180,
        "B": lambda angle: angle > 80 and angle < 180,
        "T": lambda angle: angle > 0 and angle < 60,
        "Y": lambda angle: angle > 0 and angle < 180,
        "S": lambda angle: angle > 0 and angle < 180,
        "W": lambda angle: angle > 0 and angle < 180,
        "P": lambda angle: angle > 150 and angle < 180,
        # "Move on": lambda angle: angle > 150,
    }

    if conditions.get(object_name, lambda x: False)(angle):
        data_final=object_name
    else:
        data_final='N'
    

    if hands == 'RIGHT':
        if object_name== 'P' and distance_whr is not None and shoulder_R is not None and pbox_cx is not None:
            if pbox_cx>shoulder_R :
                data_final='R'
            else:
                data_final='L'
        # else:
        #     data_final='N'
            
    elif hands == 'LEFT':
        if object_name== 'P' and distance_whl is not None  and shoulder_L is not None and pbox_cx is not None:
            if  pbox_cx>shoulder_L :
                data_final='R'
            else:
                data_final='L'
        # else:
        #     data_final='N'
            
    pbox_cx=None
    print_count+=1
    
    if print_count>data_window_size and data_final != 'N':
    # if print_count>data_window_size:
        print(data_final,angle)
        # print(data_final)

        data_final='N'
        # time2 = time.time()
        # print(f"FPS : {1 / (time2 - time1):.2f}")
        # ser.write(str(data_final).encode('utf-8'))
        print_count=0
    
    
    # print(object_name)
    ser.write(str(data_final).encode('utf-8'))
    cv2.imshow("predict", pose_color_image)  # 주석 처리된 부분은 필요에 따라 활성화할 수 있습니다.
    
    # cv2.imshow("color_image", color_image)  # 주석 처리된 부분은 필요에 따라 활성화할 수 있습니다.

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        ser.close()

pipeline.stop()  # 카메라 파이프라인을 종료합니다.
