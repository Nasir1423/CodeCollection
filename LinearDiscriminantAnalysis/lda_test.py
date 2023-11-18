""" 对 LDA 二分类投影及分类器进行了一个简单的实现 """

import numpy as np
import matplotlib.pyplot as plt

D1 = np.array([[-0.4, 0.58, 0.089],
               [-0.31, 0.27, -0.04],
               [-0.38, 0.055, -0.035],
               [-0.15, 0.53, 0.011],
               [-0.35, 0.47, 0.034],
               [0.17, 0.69, 0.1],
               [-0.011, 0.55, -0.18],
               [-0.27, 0.61, 0.12],
               [-0.065, 0.49, 0.0012],
               [-0.12, 0.054, -0.063]]).T

D2 = np.array([[0.83, 1.6, -0.014],
               [1.1, 1.6, 0.48],
               [-0.44, -0.41, 0.32],
               [0.047, -0.45, 1.4],
               [0.28, 0.35, 3.1],
               [-0.39, -0.48, 0.11],
               [0.34, -0.079, 0.14],
               [-0.3, -0.22, 2.2],
               [1.1, 1.2, -0.46],
               [0.18, -0.11, -0.49]]).T

# 原样本空间可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(D1[0, :], D1[1, :], D1[2, :], c='r', marker='^')
ax.scatter(D2[0, :], D2[1, :], D2[2, :], c='g', marker='*')

ax.set_xlabel("dimension1")
ax.set_ylabel("dimension2")
ax.set_zlabel("dimension3")
ax.set_title("samples visualization")

ax.view_init(azim=45)

plt.show()

# 1.解均值向量
mean_D1 = np.mean(D1, axis=1).reshape(-1, 1)
mean_D2 = np.mean(D2, axis=1).reshape(-1, 1)

# 2.解类内离散度矩阵
S1 = np.dot((D1 - mean_D1), (D1 - mean_D1).T)
S2 = np.dot((D2 - mean_D2), (D2 - mean_D2).T)
Sw = S1 + S2

# 3.解最佳投影向量
w = np.dot(np.linalg.inv(Sw), (mean_D1 - mean_D2))

# 4.求解分类阈值
w0 = (np.dot(w.T, mean_D1) + np.dot(w.T, mean_D2)) / 2

# 5.获取训练样本的投影结果
D1to1 = np.dot(w.T, D1)
D2to1 = np.dot(w.T, D2)

# 原样本投影结果的可视化
plt.scatter(D1to1, np.zeros(D1to1.shape), c="red")
plt.scatter(D2to1, np.zeros(D2to1.shape), c="green")
plt.title("projected samples visualization")
plt.show()

# 6.判断新数据的类别
x1 = np.array([[-0.7, 0.58, 0.089]]).T
x2 = np.array([[0.047, -0.4, 1.04]]).T

y1 = np.dot(w.T, x1)
y2 = np.dot(w.T, x2)

class_x1 = "第一类" if y1 > w0 else "第二类"
class_x2 = "第一类" if y2 > w0 else "第二类"

# 对新的数据预测的结果进行可视化
plt.scatter(D1to1, np.zeros(D1to1.shape), c="red")
plt.scatter(D2to1, np.zeros(D2to1.shape), c="green")
plt.scatter(y1, 0, c="red" if class_x1 == "第一类" else "green", marker="^")
plt.scatter(y2, 0, c="red" if class_x2 == "第一类" else "green", marker="^")
plt.title("new samples predict visualization")
plt.show()

# 各项结果打印
print("D1=", D1)
print("D2=", D2)
print("m1=", mean_D1)
print("m2=", mean_D2)
print("S1=", S1.round(4))
print("S2=", S2.round(4))
print("Sw=", Sw.round(4))
print("w*=", w.round(4))
print("w0=", w0.round(4))
print("D1投影值=", D1to1.round(4))
print("D2投影值=", D2to1.round(4))
print("测试样本 x1=", x1)
print("测试样本 x2=", x2)
print("y1=", y1)
print("y2=", y2)
print("测试样本 x1 的类别为", class_x1)
print("测试样本 x2 的类别为", class_x2)
