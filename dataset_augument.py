import cv2
import albumentations as A

# 이미지 로드
image_path = r"C:\Users\eofeh\Desktop\final_data\test\images\backward_another_p1_circle_left010.jpg"
image = cv2.imread(image_path)

# YOLO 형식의 라벨 값
class_id, cx, cy, w, h = 4, 0.472008, 0.497031, 0.062016, 0.058312

# 이미지의 실제 너비와 높이를 사용
image_height, image_width, _ = image.shape

# YOLO 형식에서 albumentations 형식으로 변환
x_min = (cx - w / 2) * image_width
y_min = (cy - h / 2) * image_height
x_max = (cx + w / 2) * image_width
y_max = (cy + h / 2) * image_height

# albumentations의 bbox 형식 (bboxes는 리스트의 리스트로 되어야 함)
bboxes = [[x_min, y_min, x_max, y_max, class_id]]  # class_id도 포함

# 증강을 정의 (여기서는 수평 뒤집기)
transform = A.Compose([
    A.HorizontalFlip(p=1.0),  # p=1.0은 항상 적용
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# 증강 적용 (class_labels 라벨 리스트를 추가로 전달)
transformed = transform(image=image, bboxes=bboxes, class_labels=[class_id])
transformed_image = transformed['image']
transformed_bboxes = transformed['bboxes']

# 결과 이미지와 바운딩 박스 시각화
for bbox in transformed_bboxes:
    x_min, y_min, x_max, y_max = bbox[:4]  # class_id는 시각화에 사용되지 않음
    cv2.rectangle(transformed_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
cv2.imwrite('transformed_image.jpg', transformed_image)
# cv2.imshow('Transformed Image', transformed_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
