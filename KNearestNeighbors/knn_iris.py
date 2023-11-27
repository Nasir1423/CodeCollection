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
