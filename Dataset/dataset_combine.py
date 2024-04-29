import os
import shutil

# 원본 폴더와 대상 폴더 경로 설정
source_folder = '../new_dataset/p8'
target_folder = os.path.join(source_folder, 'dataset')  # Simplified path

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
                    target_path = os.path.join(target_folder, f"{base}_{counter}{extension}")
                    counter += 1

            # 파일 복사 (원본에서 대상으로)
            shutil.copy(source_path, target_path)

print("모든 이미지 파일이 dataset 폴더로 복사되었습니다.")
