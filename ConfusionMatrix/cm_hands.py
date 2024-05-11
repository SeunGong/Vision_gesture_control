import numpy as np
import seaborn as sns
import supervision as sv
import matplotlib.pyplot as plt

from ultralytics import YOLO


dataset = sv.DetectionDataset.from_yolo(r'C:\Users\eofeh\Desktop\Model\datasets\test\images',r'C:\Users\eofeh\Desktop\Model\datasets\test\labels',r'C:\Users\eofeh\Desktop\Model\1.YOLOv8\yolo-combine\Train\data.yaml')

model = YOLO(r'C:\Users\eofeh\Desktop\Model\1.YOLOv8\yolo-combine\Predict\240503.pt')
def callback(image: np.ndarray) -> sv.Detections:
    result = model(image, conf=0.8, verbose=False)[0]
    # result = model(image, verbose=False)[0]
    return sv.Detections.from_ultralytics(result)

confusion_matrix = sv.ConfusionMatrix.benchmark(
   dataset = dataset,
   callback = callback
)
# confusion_matrix1 = sv.ConfusionMatrix.plot(
#    save_path=r'C:\Users\eofeh\Desktop\Model\1.YOLOv8\yolo-combine\Predict',
#    title='test',
#    normalize=False,
#    fig_size=(12, 10)
# )
classes=confusion_matrix.classes,
# classis = ['STOP','YOU','TURN', 'FORWARD', 'BACKWARD', 'POINTING']
# cm=confusion_matrix.matrix
reordered_matrix=confusion_matrix.matrix
# reordered_matrix=np.transpose(reordered_matrix)
print(classes)
print(reordered_matrix,'\n')
# cm = cm[:-1, :-1]

# print(cm)
# # 새로운 순서
new_order = [0, 3, 4, 2, 5, 1,6]

# # 행렬의 행 재배열
reordered_matrix = reordered_matrix[new_order, :][:, new_order]

# # 결과 출력
print("new order")
print(reordered_matrix)

# # 각 열의 합으로 나누어 정규화
T_1_normalized_by_columns = reordered_matrix / reordered_matrix.sum(axis=0, keepdims=True)
T_1_normalized_by_columns = np.around(T_1_normalized_by_columns, decimals=2)

print("Normalized Confusion Matrix (by columns):")
# # 출력 포맷을 대괄호로 묶고 소수점 두 자리까지 표시
for row in T_1_normalized_by_columns:
# for row in reordered_matrix:
    formatted_row = "[" + ", ".join(format(x, ".2f") for x in row) + "]"
    print(formatted_row)

fig, ax = plt.subplots( figsize=(6,6) )
# emotionlabels = ['Stop/Waving', 'YOU', 'TURN', 'FORWARD', 'BACKWARD','POINTING','background']
emotionlabels = ['Stop/Waving', 'Forward', 'Backward', 'Turn', 'Pointing','You','Background']
# sns.heatmap(reordered_matrix,
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
plt.savefig('cm(hands).png', bbox_inches = 'tight', pad_inches=0)
plt.show()