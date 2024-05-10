import os
import cv2
import shutil
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ultralytics import YOLO
from sklearn.metrics import confusion_matrix

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

source_folder = r"C:\Users\eofeh\Desktop\Model\datasets\valid\images"
folder_path  = r"C:\Users\eofeh\Desktop\Model\datasets\valid\labels"

predict_folder = os.path.join(source_folder, '../Predicted')
mismatch_folder = os.path.join(source_folder, '../Mismatch_image')  # 불일치 이미지를 저장할 폴더

# 폴더 생성을 위한 함수
def create_folder(folder_path):
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder created: {folder_path}")
        else:
            print(f"Folder already exists: {folder_path}")
    except Exception as e:
        print(f"Failed to create folder {folder_path}. Error: {str(e)}")

create_folder(predict_folder)
create_folder(mismatch_folder)

# 지원하는 이미지 확장자 목록
image_extensions = ['.jpg', '.png', '.jpeg']  # 이미지 파일 확장자

labels=[]
gestures=[]
file_names=[]

for filename in os.listdir(folder_path):
    # 파일 확장자가 .txt인 경우
    if filename.endswith('.txt'):
        # 파일 전체 경로 구성
        file_path = os.path.join(folder_path, filename)
        
        # 파일 열기
        with open(file_path, 'r') as file:
            # 파일에서 첫 번째 줄 읽기
            first_line = file.readline().strip()
            
            # 첫 번째 줄의 첫 번째 값 저장 (공백으로 구분된 경우)
            label = first_line.split()[0]
            labels.append(int(label))
            
for subdir, dirs, files in os.walk(source_folder):
    # 현재 순회 중인 폴더가 target_folder면 건너뛰기
    if subdir.startswith(predict_folder):
        continue

    for file in files:
        # 파일 확장자가 이미지 확장자 목록에 있는지 확인
            
        if any(file.lower().endswith(ext) for ext in image_extensions):
            # 이미지 파일의 원본 경로
            source_path = os.path.join(subdir, file)
            # 이미지 파일의 새 경로 (대상 폴더)
            target_path = os.path.join(predict_folder, file)

            # 파일명 충돌 방지
            if os.path.exists(target_path):
                base, extension = os.path.splitext(file)
                counter = 1
                while os.path.exists(target_path):
                    target_path = os.path.join(
                        predict_folder, f"{base}_{counter}{extension}")
                    counter += 1

            image = cv2.imread(source_path)
            color_image = np.asanyarray(image)
            
        #predict----------------------------------------------------------
            
            #MACRO
            count_gesture = 0

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

            #Reset variable 
            box_cx, box_cy = None, None  # hands box center
            cur_cx, cur_cy = None, None
            # pbox_cx, pbox_cy = None, None  # pointing box center
            lsx, lsy,rsx, rsy = None, None, None, None  # shoulder x,y
            lhy, rhy = None, None  # hip y
            euclidean_whl, euclidean_whr = None, None  # Distance both side winkle-hands
            sb_sub,sh_sub=None,None

            arm_angle = None
            arm_ratio=None

            active_hand = None
            shape_hand = None
            ratio_hand = 'N'
            final_hand = 'N'

            array_keypoints = np.zeros((keypoints_count, 2))  # [RS,RE,RW,LS,LE,LW,]
            
            results_hands = model_hands(color_image, conf=0.8, verbose=False)  
            if results_hands is not None:
                for r in results_hands:
                    boxes = r.boxes  # Boxes class
                    
                    #Check front hand
                    for number_box, box in enumerate(boxes):  
                        b = box.xyxy[0].to('cpu').detach().numpy().copy()
                        x1, y1, x2, y2 = map(int, b[:4])# Box left top and right bottom coordinate
                        box_cx, box_cy = int((x2 - x1) / 2 + x1), int((y2 - y1) / 2 + y1)
                        shape_hand = model_hands.names[int(box.cls)]

                        cv2.rectangle(color_image, (x1, y1), (x2, y2),(0, 0, 255), thickness=2, lineType=cv2.LINE_4)
                        cv2.putText(color_image, shape_hand, (box_cx, box_cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                        shape_hand = shape_hand[0]

            #################### Predict pose ####################
            results_pose = model_pose(color_image, conf=0.8, verbose=False)  # Predict pose
            pose_color_image = results_pose[0].plot()  # Draw skelton to pose image

            if results_pose is not None:
                for r in results_pose:
                    keypoints = r.keypoints
                    pose_boxes = r.boxes
                    
                    min_pose_depth = float('inf')
                    final_pose_index = 0

                    #Check front pose
                    for number_pose_box, box in enumerate(pose_boxes): 
                        b = box.xyxy[0].to('cpu').detach().numpy().copy()
                        px1, py1, px2, py2 = map(int, b[:4])

                    #Check out of boundary box
                    if (box_cy < py1 or box_cy > py2 or box_cx < px1 or box_cx > px2):
                        continue
                        
                    #Get keypoints
                    for i, k in enumerate(keypoints):
                        for kp_index, ap_index in keypoint_indices.items():
                            if k.xy[0].size(0) > kp_index:
                                array_keypoints[ap_index] = k.xy[0][kp_index].cpu().numpy()
                    #put value in variable
                    lsx = int(array_keypoints[3][0]) 
                    lsy = int(array_keypoints[3][1])  
                    rsx = int(array_keypoints[0][0])
                    rsy = int(array_keypoints[0][1])
                    rhy = int(array_keypoints[7][1])
                    lhy = int(array_keypoints[8][1])
            ###################################################################################

            #Distinction between left and right hands
            if box_cx is not None and box_cy is not None:
                euclidean_whr = np.sqrt((box_cx - int(array_keypoints[2][0]))**2 + (box_cy - int(array_keypoints[2][1]))**2)
                euclidean_whl = np.sqrt((box_cx - int(array_keypoints[5][0]))**2 + (box_cy - int(array_keypoints[5][1]))**2)
                
                # Activate hand selection
                if euclidean_whl is not None and euclidean_whr is not None:  
                    if (euclidean_whl > euclidean_whr):
                        active_hand = 'RIGHT'
                        arm_angle = calculate_angle_arm(array_keypoints[0], array_keypoints[1], array_keypoints[2])
                    elif (euclidean_whl < euclidean_whr):
                        active_hand = 'LEFT'
                        arm_angle = calculate_angle_arm(array_keypoints[3], array_keypoints[4], array_keypoints[5])

                    # Get ratio between shoulder-hip and shoulder-box
                    if (active_hand == 'RIGHT' and rhy is not None and rsy is not None):
                        if(rhy>0 and rsy>0):
                            sh_sub = rhy-rsy
                            sb_sub = abs(box_cy-rsy)
                            arm_ratio=sb_sub/sh_sub
                    elif (active_hand == 'LEFT' and lhy is not None and lsy is not None):
                        if(lhy>0 and lsy>0):
                            sh_sub = lhy-lsy
                            sb_sub = abs(box_cy-lsy)
                            arm_ratio=sb_sub/sh_sub
                            
                    #Check arm_ratio and arm_angle
                    if(shape_hand=='S'):
                        ratio_hand=shape_hand
                    elif(arm_ratio is not None and arm_ratio<0.3):
                        if(shape_hand=='T'):
                            ratio_hand=shape_hand
                        elif(shape_hand=='Y'):  
                            ratio_hand=shape_hand
                    elif(arm_ratio is not None and arm_ratio>0.45):
                        if(shape_hand=='F'):
                            ratio_hand=shape_hand
                        elif(shape_hand=='B'and arm_angle<120):
                            ratio_hand=shape_hand
                        elif(shape_hand=='P'):
                            ratio_hand=shape_hand

                gestures.append(ratio_hand) #add predicted gesture to gestures array
                file_names.append(file)
                cv2.imwrite(target_path, color_image)
        #predict----------------------------------------------------------
                       
pred_labels = np.array(gestures)  # 모델과 post-processing을 통해 얻은 예측 결과
true_labels = np.array(labels)

with open('mismatch.txt', 'w') as file:
    for index, (pred_label, true_label) in enumerate(zip(pred_labels, true_labels)):
        if pred_label != true_label:
            file.write(f"{index}: {file_names[index]}: {pred_label} != {true_label}\n")

            # 불일치하는 이미지를 별도 폴더에 저장
            source_path = os.path.join(predict_folder, file_names[index])  # 이미지 파일 경로 재구성
            target_mismatch_path = os.path.join(mismatch_folder, f"mismatch_{file_names[index]}")
            shutil.copy(source_path, target_mismatch_path)

"""draw confusion matrix
cm = confusion_matrix(true_labels, pred_labels, labels=[0,3,4,2,5,1,6])

print(cm)
cm=np.transpose(cm)

# 각 열의 합으로 나누어 정규화
T_1_normalized_by_columns = cm / cm.sum(axis=0, keepdims=True)
T_1_normalized_by_columns = np.around(T_1_normalized_by_columns, decimals=2)
print("Normalized Confusion Matrix (by columns):")

# 출력 포맷을 대괄호로 묶고 소수점 두 자리까지 표시
for row in T_1_normalized_by_columns:
    formatted_row = "[" + ", ".join(format(x, ".2f") for x in row) + "]"
    print(formatted_row)

fig, ax = plt.subplots( figsize=(6,6) )
emotionlabels = ['Stop/Waving', 'Forward', 'Backward', 'Turn', 'Pointing','You','Background']
sns.heatmap(T_1_normalized_by_columns,
            cmap = 'Blues',
            annot = True,   # 실제 값을 표시한다
            linewidths=.5,  # 경계면 실선으로 구분하기
            vmin = 0,vmax = 1,
            xticklabels = emotionlabels,
            yticklabels = emotionlabels,
           )
plt.xlabel('True')
plt.ylabel('Predict')
plt.xticks(rotation=45, ha='right')  # Rotate labels and align them horizontally
plt.yticks(rotation=0)
ax.tick_params(axis='both', which='major', labelsize=12)  # Change label size
plt.tight_layout()
plt.savefig('cm(integrated).png', bbox_inches = 'tight', pad_inches=0)
plt.show()
"""