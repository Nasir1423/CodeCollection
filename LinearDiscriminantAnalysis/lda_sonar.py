import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from LinearDiscriminantAnalysis import LinearDiscriminantAnalysis
import pandas as pd

# 加载数据
sonar = pd.read_csv("sonar.csv")
sonar_data = np.asarray(sonar.iloc[:, :-1])
sonar_target = np.asarray(sonar.iloc[:, -1].replace({"M": 1, "R": 0}).astype(int))

# 数据预处理
scaler = StandardScaler()
sonar_data = scaler.fit_transform(sonar_data)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(sonar_data, sonar_target, test_size=0.25)

# 模型训练
clf = LinearDiscriminantAnalysis(x_train, y_train)
clf.fit()

# 模型测试：二维可视化数据、模型预测精度输出
clf.visualize(x_test, y_test)
score = clf.score(x_test, y_test)
print(f"精确度为 {score * 100:.2f}%")  # 80.77%
print(f"唢呐数据集的降维后维数为 {clf.projectMatrix.shape[1]}")  # 1
