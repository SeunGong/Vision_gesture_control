import cv2 #OpenCV library
import sys #for accessing system variable
import os #file paths
import time #measuring time
import numpy as np 
import pyrealsense2 as rs # Library for controlling and obtaining data from Intel RealSense cameras.
from ultralytics import YOLO #Imports the YOLOv8 object detection implementation from Ultralytics.

#Define the desired width and height of the camera frames.
W = 640
H = 480

#Initialize the RealSense camera pipeline
config = rs.config()
#starts the color and depth video stream
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
# config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)

pipeline = rs.pipeline() #managing the RealSense data Start and stop camera stream, Retrieve image frames, Apply image processing filters or transformations
profile = pipeline.start(config) #profile contaions important information about the active stremas and connected device metadata

#Setting up alignment of the color and depth image streams.
align_to = rs.stream.color
align = rs.align(align_to)

#Loads YOLOv8 model file from the path.
# model_directory = os.environ['HOME'] + '/yolov8_rs/yolov8m.pt'
# model = YOLO(model_directory)

model_pose = YOLO("yolov8m-pose")
model_hands = YOLO("best.pt")
# result = model(source=0, show=True, conf=0.7 ,save=True)

while True: #Continuously processes frames from the camera
    time1 = time.time()
    frames = pipeline.wait_for_frames() #Retrieves color and depth image frames from the RealSense

    aligned_frames = align.process(frames)#Aligns the depth frame to match the color frame's perspective.
    color_frame = aligned_frames.get_color_frame()
    # depth_frame = aligned_frames.get_depth_frame()
    #Skips the frame if there's no color image data.
    if not color_frame:
        continue
    
    #Converts the frame data into NumPy arrays for easy manipulation.
    color_image = np.asanyarray(color_frame.get_data())
    # depth_image = np.asanyarray(depth_frame.get_data())
    
    #Applies a color map to the depth image for better visualization.
    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)

    #Runs the YOLOv8 model on the color image to get detection results.
    results_hands = model_hands(color_image,conf=0.8)
    # cropped_image=color_image
    for r in results_hands:
        boxes = r.boxes
        names = r.names
        print(names)
        for box in boxes:
            b = box.xyxy[0].to('cpu').detach().numpy().copy()
            c = box.cls 

            # 1. Bounding Box Cropping
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            # cropped_image = color_image[y1:y2, x1:x2]
            cv2.rectangle(color_image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255),
                          thickness = 2, lineType=cv2.LINE_4)
            cv2.putText(color_image, text = model_hands.names[int(c)], org=(int(b[0]), int(b[1])),
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.7, color = (0, 0, 255),
                        thickness = 2, lineType=cv2.LINE_4)
            # 2. Pose Estimation on Cropped Image
            
    
    # Predict image using pose model
    # results_pose = model_pose(color_image, conf=0.8)
    # annotated_frame = results_pose[0].plot()

    
    #Displays both the color image with annotations and the colorized depth image.
    # cv2.imshow("color_image", annotated_frame)
    cv2.imshow("color_image", color_image)
    # cv2.imshow("depth_image", depth_colormap)
    time2 = time.time()
    print(f"FPS : {int(1/(time2-time1))}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
         break
