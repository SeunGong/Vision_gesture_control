import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#hand+pose
# T1=[[1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
# [0.00, 1.00, 0.01, 0.00, 0.00, 0.00],
# [0.00, 0.00, 0.99, 0.00, 0.00, 0.00],
# [0.00, 0.00, 0.00, 1.00, 0.00, 0.00],
# [0.00, 0.00, 0.00, 0.00, 0.92, 0.00],
# [0.00, 0.00, 0.00, 0.00, 0.08, 1.00]]

#hand+pose
T1=[[1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
[0.00, 1.00, 0.00, 0.00, 0.00, 0.00],
[0.00, 0.00, 1.00, 0.00, 0.00, 0.00],
[0.00, 0.00, 0.00, 1.00, 0.00, 0.00],
[0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
[0.00, 0.00, 0.00, 0.00, 0.00, 1.00]]

fig, ax = plt.subplots( figsize=(6,6) )
# gesture_labels = ['Stop/Waving', 'YOU', 'TURN', 'FORWARD', 'BACKWARD','POINTING','background']
gesture_labels = ['Stop/Waving', 'Forward', 'Backward', 'Turn', 'Pointing','You']
# sns.heatmap(reordered_matrix,
sns.heatmap(T1,
            cmap = 'Blues',
            annot = True,   # 실제 값을 표시한다
            linewidths=.5,  # 경계면 실선으로 구분하기
            vmin = 0,vmax = 1,
            xticklabels = gesture_labels,
            yticklabels = gesture_labels,
           )
plt.xlabel('True')
plt.ylabel('Predict')
plt.xticks(rotation=45, ha='right')  # Rotate labels and align them horizontally
plt.yticks(rotation=0)
ax.tick_params(axis='both', which='major', labelsize=12)  # Change label size
plt.tight_layout()
# plt.savefig('Confusion_matrix(hand).png', bbox_inches = 'tight', pad_inches=0)
plt.savefig('Confusion_matrix(hand+pose).png', bbox_inches = 'tight', pad_inches=0)
plt.show()