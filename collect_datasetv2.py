import cv2
import pyrealsense2 as rs
import numpy as np
import time
import os

# RealSense pipeline 구성
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 카메라 스트리밍 시작
pipeline.start(config)

# 윈도우 이름
window_name = 'RealSense Image'

# 사용자 입력에 따라 순차적으로 변경될 설정들
actions = ['you','you_circle', 'stop', 'forward', 'backward', 'turn','turn_circle', 'pointing']
directions = ['_left', '_right']
user='p1'
current_action_index = 0
current_direction_index = 0

try:
    while True:
        # 현재 설정에 따른 save_path 설정
        current_action = actions[current_action_index]
        current_direction = directions[current_direction_index]
        save_path = f"{current_action}"

        # 폴더 확인 및 생성
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 100장 이미지 캡처 및 저장
        for i in range(10):
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame:
                color_image = np.asanyarray(color_frame.get_data())

                # 이미지 저장
                filename = f"{save_path}/{save_path}_{current_direction}_{user}_{i:03d}.jpg"
                cv2.imwrite(filename, color_image)

                # 이미지 디스플레이
                text = f"{i}"
                cv2.putText(color_image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow(window_name, color_image)

                # 'q'를 누르면 종료
                if cv2.waitKey(1) == ord('q'):
                    raise KeyboardInterrupt

                time.sleep(0.1)

        # 다음 설정으로 업데이트
        current_direction_index += 1
        if current_direction_index >= len(directions):
            current_direction_index = 0
            current_action_index += 1
            if current_action_index >= len(actions):
                print("모든 작업이 완료되었습니다.")
                break

        # 사용자 입력 대기 ('Enter' 키를 누르면 다음으로 넘어감)
        print("Next : "f"{actions[current_action_index]}{directions[current_direction_index]}")
        input("다음 설정으로 진행하려면 Enter를 누르세요...")

finally:
    # 자원 해제
    pipeline.stop()
    cv2.destroyAllWindows()
