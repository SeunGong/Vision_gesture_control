# import os
# import cv2
# import shutil
# from ultralytics import YOLO

# # Load a model
# # model = YOLO('yolov8n.pt')  # load an official model
# model = YOLO(r"C:\Users\eofeh\Desktop\Model\1.YOLOv8\yolo-combine\Predict\240501.pt")  # load a custom model

# # Predict with the model
# # results = model(r"C:\Users\eofeh\Desktop\Hand Shape Data\Data(image)\data(3floor)\turn_extend\turn_extend_right035.jpg")  # predict on an image


# # 원본 폴더와 대상 폴더 경로 설정
# source_folder = r"C:\Users\eofeh\Desktop\Hand Shape Data\Data(image)\data(1floor)"
# target_folder = os.path.join(source_folder, 'Model_predict')  # Simplified path

# # 대상 폴더가 없으면 생성
# if not os.path.exists(target_folder):
#     os.makedirs(target_folder)

# # 지원하는 이미지 확장자 목록
# image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

# # source_folder 내의 모든 하위 폴더 탐색, 단 target_folder는 제외
# for subdir, dirs, files in os.walk(source_folder):
#     # 현재 순회 중인 폴더가 target_folder면 건너뛰기
#     if subdir.startswith(target_folder):
#         continue

#     for file in files:
#         # 파일 확장자가 이미지 확장자 목록에 있는지 확인
#         if any(file.lower().endswith(ext) for ext in image_extensions):
#             # 이미지 파일의 원본 경로
#             source_path = os.path.join(subdir, file)
#             # 이미지 파일의 새 경로 (대상 폴더)
#             target_path = os.path.join(target_folder, file)

#             # 파일명 충돌 방지
#             if os.path.exists(target_path):
#                 base, extension = os.path.splitext(file)
#                 counter = 1
#                 while os.path.exists(target_path):
#                     target_path = os.path.join(target_folder, f"{base}_{counter}{extension}")
#                     counter += 1

#             # 파일 복사 (원본에서 대상으로)
#             results = model(source_path, conf=0.8, verbose=False)  # predict on an image
#             image=cv2.imread(source_path)
#             if results is not None:
#                 for r in results:
#                     boxes = r.boxes
#                     for box in boxes:
#                         # class_index = box.cls  # Get the class index of the object
#                         # Use the index to get the object's name
#                         b = box.xyxy[0].to('cpu').detach().numpy().copy()
#                         c = box.cls
#                         x1, y1, x2, y2 = map(int, b[:4])

#                         cv2.rectangle(image, (x1, y1), (x2, y2),
#                                   (0, 0, 255), thickness=2, lineType=cv2.LINE_4)
#                         cv2.putText(image,  model.names[int(c)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
#                             0.7, (0, 0, 255), 2, cv2.LINE_4)
#             shutil.copy(source_path, target_path)

# print("모든 이미지 파일에 대한 Predict를 수행하였습니다.")

import os
import cv2
import shutil
from ultralytics import YOLO

# Load a model
# load a custom model
model = YOLO(
    r"C:\Users\eofeh\Desktop\Model\1.YOLOv8\yolo-combine\Predict\240501.pt")

# 원본 폴더와 대상 폴더 경로 설정
source_folder = r"C:\Users\eofeh\Desktop\Dataset\images(me)\3floor(me)"
target_folder = os.path.join(source_folder, '../Model_predict')  # Simplified path

# 대상 폴더가 없으면 생성
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 지원하는 이미지 확장자 목록
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

# source_folder 내의 모든 하위 폴더 탐색, 단 target_folder는 제외
for subdir, dirs, files in os.walk(source_folder):
    # 현재 순회 중인 폴더가 target_folder면 건너뛰기
    if subdir.startswith(target_folder):
        continue

    for file in files:
        # 파일 확장자가 이미지 확장자 목록에 있는지 확인
        if any(file.lower().endswith(ext) for ext in image_extensions):
            # 이미지 파일의 원본 경로
            source_path = os.path.join(subdir, file)
            # 이미지 파일의 새 경로 (대상 폴더)
            target_path = os.path.join(target_folder, file)

            # 파일명 충돌 방지
            if os.path.exists(target_path):
                base, extension = os.path.splitext(file)
                counter = 1
                while os.path.exists(target_path):
                    target_path = os.path.join(
                        target_folder, f"{base}_{counter}{extension}")
                    counter += 1

            # 이미지 로드
            image = cv2.imread(source_path)
            # 모델 예측
            # predict on an image
            results = model(source_path, conf=0.8, verbose=False)
            if results is not None:
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        b = box.xyxy[0].to('cpu').detach().numpy().copy()
                        c = box.cls
                        x1, y1, x2, y2 = map(int, b[:4])
                        cv2.rectangle(image, (x1, y1), (x2, y2),
                                      (0, 0, 255), thickness=2, lineType=cv2.LINE_4)
                        cv2.putText(image,  model.names[int(c)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 255), 2, cv2.LINE_4)
                        # cv2.rectangle(image, (x1, y1), (x2, y2),
                        #               (0, 0, 255), thickness=2, lineType=cv2.LINE_4)
                        # cv2.putText(
                        #     image, model.names[c], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 수정된 이미지 저장
            cv2.imwrite(target_path, image)

print("모든 이미지 파일에 대한 Predict를 수행하였습니다.")
