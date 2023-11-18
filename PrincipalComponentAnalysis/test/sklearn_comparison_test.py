from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

iris_dataset = load_iris()
iris_target = iris_dataset['target']  # 数据标签
iris_target_names = iris_dataset['target_names']  # 数据标签名
iris_data = iris_dataset['data']  # 数据集
iris_feature_names = iris_dataset['feature_names']  # 数据集特征名

pca = PCA()
iris_data_projected = pca.fit_transform(iris_data)

# 二维可视化
# 获取样本的类别情况
unique_labels = np.unique(iris_target)

# 将数据和标签拼接
target = iris_target.reshape(-1, 1)
data = np.concatenate((iris_data_projected, target), axis=1)

# 替换为系统中存在的中文字体
plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'

for label in unique_labels:
    class_points = data[data[:, -1] == label]  # 获取类别为 label 的样本点
    plt.scatter(class_points[:, 0], class_points[:, 1], label=f"Class {label}")

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("主成分分析可视化散点图")
plt.legend()
plt.show()
