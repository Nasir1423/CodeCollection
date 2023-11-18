""" 基于 "sonar 数据集" 及 "线性判别分析模型" 对 "PCA 降维效果" 进行分析及可视化 """
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from DimensionalityReduction.PCA import PCA
import pandas as pd
import matplotlib.pyplot as plt

# 替换为系统中存在的中文字体
plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'

# 载入 sonar 数据集
# 类别 M 代表 Mines，类别 R 代表 Rocks；共 208 个样本，每个样本 60 个特征和 1 个类别标签
sonar_dataset = pd.read_csv("sonar.csv", header=None)
sonar_data = sonar_dataset.iloc[:, :-1]
sonar_target = sonar_dataset.iloc[:, -1].replace({"M": 1, "R": 0}).astype(int)

# 获取 PCA 降维后的数据集
pca = PCA(sonar_data)
sonar_data_pca = pca.project()

# 划分数据集（对原始数据和降维后的数据分别进行划分）
x_train, x_test, y_train, y_test = train_test_split(sonar_data, sonar_target, test_size=0.25, random_state=42)
x_train_pca, x_test_pca, y_train_pca, y_test_pca = train_test_split(sonar_data_pca, sonar_target, test_size=0.25,
                                                                    random_state=42)

# 情形 1：直接利用原始数据集进行模型训练
clf1 = LinearDiscriminantAnalysis()
clf1.fit(x_train, y_train)
score1 = clf1.score(x_test, y_test)
print(f"未经 PCA 降维的 sonar 数据集训练得到的 LDA 模型的准确度为: {score1 * 100:.2f}%")

# # 对 LDA transform 的数据进行可视化（未 PCA 降维）
# X1 = clf1.transform(sonar_data)
# plt.scatter(X1, np.zeros(X1.shape), c=sonar_target)
# plt.title("sonar 数据集 LDA transformed data 的可视化 (PCA=False)")
# plt.show()

# 情形 2：利用降维后的数据进行模型训练
clf2 = LinearDiscriminantAnalysis()
clf2.fit(x_train_pca, y_train_pca)
score2 = clf2.score(x_test_pca, y_test_pca)
print(f"经过 PCA 降维的 sonar 数据集训练得到的 LDA 模型的准确度为: {score2 * 100:.2f}%")
print(f"经过 PCA 降维，LDA 模型在 sonar 数据集上的准确度提升了 {((score2 - score1) / score1) * 100:.2f}%")

# 对 LDA transform 的数据进行可视化（PCA 降维）
X2 = clf2.transform(sonar_data_pca)
plt.scatter(X2, np.zeros(X2.shape), c=sonar_target)
plt.title("sonar 数据集 LDA transformed data 的可视化 (PCA=True)")
plt.show()

# # PCA 结果可视化
pca.visualize(sonar_target)
