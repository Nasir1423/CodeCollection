import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from KNearestNeighbors import KNearestNeighbors
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

clf = KNearestNeighbors(x_train, y_train, n_neighbors=11)
score = clf.score(x_test, y_test)
print(f"KNN(K=11) 在 SONAR 上的精确度为 {score * 100:.2f}%")  # 75.00%

clf.n_neighbors = 7
score = clf.score(x_test, y_test)
print(f"KNN(K=7) 在 SONAR 上的精确度为 {score * 100:.2f}%")  # 78.85%

clf.n_neighbors = 5
score = clf.score(x_test, y_test)
print(f"KNN(K=5) 在 SONAR 上的精确度为 {score * 100:.2f}%")  # 80.77%

clf.n_neighbors = 1
score = clf.score(x_test, y_test)
print(f"KNN(K=1) 在 SONAR 上的精确度为 {score * 100:.2f}%")  # 90.38%
