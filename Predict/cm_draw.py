import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# T1=[[1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0],
# [0.00, 0.99, 0.00, 0.01, 0.00, 0.00, 0],
# [0.00, 0.01, 1.00, 0.00, 0.00, 0.00, 0],
# [0.00, 0.00, 0.00, 0.99, 0.00, 0.00, 0],
# [0.00, 0.00, 0.00, 0.00, 1.00, 0.10, 0],
# [0.00, 0.00, 0.00, 0.00, 0.00, 0.90, 0],
# [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0]]

T1=[[1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.33],
[0.00, 0.99, 0.00, 0.01, 0.00, 0.00, 0.00],
[0.00, 0.01, 1.00, 0.00, 0.00, 0.00, 0.00],
[0.00, 0.00, 0.00, 0.99, 0.00, 0.00, 0.33],
[0.00, 0.00, 0.00, 0.00, 1.00, 0.10, 0.33],
[0.00, 0.00, 0.00, 0.00, 0.00, 0.90, 0.00],
[0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]]
fig, ax = plt.subplots( figsize=(6,6) )
# emotionlabels = ['Stop/Waving', 'YOU', 'TURN', 'FORWARD', 'BACKWARD','POINTING','background']
emotionlabels = ['Stop/Waving', 'Forward', 'Backward', 'Turn', 'Pointing','You','Background']
# sns.heatmap(reordered_matrix,
sns.heatmap(T1,
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
plt.savefig('cm(hand).png', bbox_inches = 'tight', pad_inches=0)
plt.show()