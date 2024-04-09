import os
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

# 이미지 데이터가 있는 폴더와 대상 폴더 지정
source_folder = '/path/to/your/images'
train_folder = '/path/to/your/train'
valid_folder = '/path/to/your/valid'
test_folder = '/path/to/your/test'

# 소스 폴더에서 이미지 파일 리스트 생성
image_files = [f for f in os.listdir(source_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# 파일 목록을 훈련(70%), 나머지(30%)로 분할
train_files, remaining_files = train_test_split(image_files, test_size=0.3, random_state=42)

# 나머지 파일을 검증(2/3)과 테스트(1/3)로 분할
valid_files, test_files = train_test_split(remaining_files, test_size=1/3, random_state=42)

# 이미지 파일을 지정된 폴더로 복사하는 함수
def copy_files(files, source_folder, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    for file in files:
        shutil.copy(os.path.join(source_folder, file), os.path.join(target_folder, file))

# 분류된 파일을 각각의 폴더로 복사
copy_files(train_files, source_folder, train_folder)
copy_files(valid_files, source_folder, valid_folder)
copy_files(test_files, source_folder, test_folder)

print("이미지 분류 완료.")
