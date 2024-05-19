import numpy as np

def calculate_arm_angle(a, b, c):

    # 각 값을 받아 넘파이 배열로 변형
    a = np.array(a)  # 첫번째
    b = np.array(b)  # 두번째
    c = np.array(c)  # 세번째

    # 라디안을 계산하고 실제 각도로 변경한다.
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    arm_angle = np.abs(radians*180.0/np.pi)

    # 180도가 넘으면 360에서 뺀 값을 계산한다.
    if arm_angle > 180.0:
        arm_angle = 360-arm_angle

    # 각도를 리턴한다.
    return arm_angle

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


def calculate_arm_ratio(box_cy, shoulder_y, hip_y):
    if hip_y > 0 and shoulder_y > 0:
        sh_sub = hip_y - shoulder_y
        sb_sub = abs(box_cy - shoulder_y)
        return sb_sub / sh_sub
    return None



# Distinction between left and right hands
def calculate_euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)



# Activate hand selection
def select_active_hand(box_cx, box_cy, keypoints):
    euclidean_whl = calculate_euclidean_distance(box_cx, box_cy, keypoints[5][0], keypoints[5][1])
    euclidean_whr = calculate_euclidean_distance(box_cx, box_cy, keypoints[2][0], keypoints[2][1])

    if euclidean_whl < euclidean_whr:
        active_hand = "LEFT"
        arm_angle = calculate_arm_angle(keypoints[3], keypoints[4], keypoints[5])
        arm_ratio = calculate_arm_ratio(box_cy, keypoints[3][1], keypoints[8][1])
    else:
        active_hand = "RIGHT"
        arm_angle = calculate_arm_angle(keypoints[0], keypoints[1], keypoints[2])
        arm_ratio = calculate_arm_ratio(box_cy, keypoints[0][1], keypoints[7][1])

    return active_hand, arm_angle, arm_ratio



def get_box_coordinates(box, depth_frame, model_hands, DEPTH_DISTANCE_MAX):
    b = box.xyxy[0].to("cpu").detach().numpy().copy()
    x1, y1, x2, y2 = map(int, b[:4])
    box_cx = int((x2 - x1) / 2 + x1)
    box_cy = int((y2 - y1) / 2 + y1)
    depth_box = depth_frame.get_distance(box_cx, box_cy)

    if depth_box > DEPTH_DISTANCE_MAX:
        depth_box = DEPTH_DISTANCE_MAX
    elif depth_box < 0:
        depth_box = 0

    shape_hand = model_hands.names[int(box.cls)]

    return (x1, y1, x2, y2, box_cx, box_cy, depth_box, shape_hand)