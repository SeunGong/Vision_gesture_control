import os
from collections import defaultdict

# 폴더 경로 리스트
folders = [r'C:\Users\eofeh\Desktop\Model\datasets\test\images',r'C:\Users\eofeh\Desktop\Model\datasets\test\labels']

# 각 파일 이름(확장자 제외)이 나타난 폴더 수를 저장하는 사전
file_name_occurrences = defaultdict(int)

# 각 폴더의 파일 이름(확장자 제외)을 저장하는 사전
folder_files = defaultdict(set)

# 각 폴더를 순회하며 파일 이름(확장자 제외) 추출 및 카운트
for folder in folders:
    for file in os.listdir(folder):
        file_name = os.path.splitext(file)[0]
        file_name_occurrences[file_name] += 1
        folder_files[folder].add(file_name)

# 공통적으로 존재하지 않는 파일 이름 찾기
uncommon_file_names = {file_name for file_name, count in file_name_occurrences.items() if count < len(folders)}

# 각 폴더에서 공통적으로 존재하지 않는 파일 삭제
for folder in folders:
    for file in os.listdir(folder):
        file_name = os.path.splitext(file)[0]
        if file_name in uncommon_file_names:
            # 파일 삭제
            os.remove(os.path.join(folder, file))
            print(f"Deleted {file} from {folder}")

print("공통적으로 존재하지 않는 파일을 모두 삭제했습니다.")
