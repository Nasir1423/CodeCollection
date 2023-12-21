from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from KNearestNeighbors import KNearestNeighbors

# 加载鸢尾花数据集
iris = load_iris()
iris_data = iris.data
iris_target = iris.target

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_target, test_size=0.25)

clf = KNearestNeighbors(x_train, y_train, n_neighbors=21)
score = clf.score(x_test, y_test)
print(f"KNN(K=21) 在 IRIS 上的精确度为 {score * 100:.2f}%")  # 92.11%

clf.n_neighbors = 7
score = clf.score(x_test, y_test)
print(f"KNN(K=7) 在 IRIS 上的精确度为 {score * 100:.2f}%")  # 94.74%

clf.n_neighbors = 5
score = clf.score(x_test, y_test)
print(f"KNN(K=5) 在 IRIS 上的精确度为 {score * 100:.2f}%")  # 94.74%

clf.n_neighbors = 1
score = clf.score(x_test, y_test)
print(f"KNN(K=1) 在 IRIS 上的精确度为 {score * 100:.2f}%")  # 94.74%

# import matplotlib.pyplot as plt
# x = []
# Y = []
# for m in range(50):
#     clf.n_neighbors = m+1
#     x.append(clf.n_neighbors)
#     score = clf.score(x_test, y_test)
#     Y.append(score)
# plt.xlabel('k')
# plt.title('Iris')
# plt.ylabel('score')
# plt.ylim((0.8, 1))  # 纵坐标的范围
# plt.plot(x, Y, color='blue', alpha=0.5, linewidth=1)
# plt.plot(x, Y, 'g*')
# plt.show()