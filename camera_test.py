import cv2
import pyrealsense2 as rs
import numpy as np # Import NumPy
import time
import os

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start camera streaming
pipeline.start(config)

# Window name
window_name = 'RealSense Image'

# Check file exist
# save_path = "test"
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
    
try:
    # Capture and save images
    # for i in range(100):
    while(True):
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame:
            color_image = np.asanyarray(color_frame.get_data())

            # Save image
            # cv2.imwrite(f"{save_path}/{save_path}_p1_circle_left{i:03d}.jpg", color_image)

            # Display image 
            text=str(i)
            cv2.putText(color_image,text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
            cv2.imshow(window_name, color_image)
            # Exit if 'q' is pressed 
        if cv2.waitKey(1) == ord('q'):
            break

        # time.sleep(0.1)
    


finally:
    # Release resources
    pipeline.stop()
    cv2.destroyAllWindows() 
