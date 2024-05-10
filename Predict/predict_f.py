import numpy as np

def calculate_angle_arm(a, b, c):

    # 각 값을 받아 넘파이 배열로 변형
    a = np.array(a)  # 첫번째
    b = np.array(b)  # 두번째
    c = np.array(c)  # 세번째

    # 라디안을 계산하고 실제 각도로 변경한다.
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle_arm = np.abs(radians*180.0/np.pi)

    # 180도가 넘으면 360에서 뺀 값을 계산한다.
    if angle_arm > 180.0:
        angle_arm = 360-angle_arm

    # 각도를 리턴한다.
    return angle_arm

# def getNoseDepth():
#     #Get nose depth
#     try: 
#         depth_nose = depth_frame.get_distance(int(array_keypoints[6][0]), int(array_keypoints[6][1]))
#         # cv2.putText(pose_color_image, f"Depth: {depth_nose}", (int(array_keypoints[6][0]), int(
#         #     array_keypoints[6][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_4)
#     except RuntimeError as e:
#         print(f"An error occurred: {e}")
#     #Get distance between nose and hand
#     if (depth_nose is not None and depth_hand is not None):
#         if (depth_nose != 0.0 and depth_hand != 0.0):
#             if (depth_nose-depth_hand < 0):
#                 continue
#             # print('{0:.2f} {1} {2}'.format(depth_nose-depth_hand, depth_nose, depth_hand))

# class Person():
#     def __init__(self,name,age):
#         self.name=name
#         self.age=age
#         self.is_alive=True
        
#     def one_year_later(self):
#         self.age +=1
    
#     def change_name(self,name):
#         self.name=name