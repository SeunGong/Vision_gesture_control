import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
T_1 = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 1.0, 0.01, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.99, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.99, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 1.0, 0.12],
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.87]]
# 그림 사이즈 지정



fig, ax = plt.subplots( figsize=(6,6) )
# 삼각형 마스크를 만든다(위 쪽 삼각형에 True, 아래 삼각형에 False)
# mask = np.zeros_like(T)
# mask[np.triu_indices_from(mask)] = True
# 히트맵을 그린다
emotionlabels = ['Stop/Waving', 'Forward', 'Backward', 'Turn', 'Pointing','You']
sns.heatmap(T_1,
            cmap = 'Blues',
            annot = True,   # 실제 값을 표시한다
            # mask=mask,      # 표시하지 않을 마스크 부분을 지정한다
            linewidths=.5,  # 경계면 실선으로 구분하기
            # cbar_kws={"shrink": .5},# 컬러바 크기 절반으로 줄이기
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
plt.savefig('confusion matrix plot.png', bbox_inches = 'tight', pad_inches=0)
plt.show()