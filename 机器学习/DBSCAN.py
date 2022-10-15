import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
# 导入数据,sklearn自带鸢尾花数据集


X=np.array([[69.33,6.32],[87.05,2.01],[61.71,5.87],[65.88,7.12],[61.58,7.35],[67.65,0.00],[59.81,5.41],[92.63,1.07],[95.02,0.62],[96.77,0.21],
            [94.29,0.72],[59.01,8.70],[62.47,8.23],[61.87,0.00],[65.18,8.27],[60.71,0.00],[79.46,0.00],[76.68,4.71],[92.35,1.66],[92.72,0.94]])#注意数组的形式


X_db = DBSCAN(eps=0.6, min_samples=4).fit_predict(X)
# 设置半径为0.6，最小样本量为2，建模
db = DBSCAN(eps=10, min_samples=2).fit(X)

# 统计每一类的数量
counts = pd.value_counts(X_db, sort=True)
print(counts)
plt.rcParams['font.sans-serif'] = [u'Microsoft YaHei']

fig,ax = plt.subplots(1,2,figsize=(12,12))

# 画聚类后的结果
ax1 = ax[0]
ax1.scatter(x=X[:,0],y=X[:,1],s=250,c=X_db)
ax1.set_title('DBSCAN聚类结果',fontsize=20)

# 画真实数据结果
ax2 = ax[1]
ax2.scatter(x=X[:,0],y=X[:,1],s=250)
ax2.set_title('真实分类',fontsize=20)
plt.show()