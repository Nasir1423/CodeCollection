from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from LinearDiscriminantAnalysis import LinearDiscriminantAnalysis

# 加载鸢尾花数据集
iris = load_iris()
iris_data = iris.data
iris_target = iris.target

# 数据预处理
# scaler = StandardScaler()
# iris_data = scaler.fit_transform(iris_data)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_target, test_size=0.25)

# 模型训练
clf = LinearDiscriminantAnalysis(x_train, y_train)
clf.fit()

# 模型测试：二维可视化数据、模型预测精度输出
clf.visualize(x_test, y_test)
score = clf.score(x_test, y_test)
print(f"精确度为 {score*100:.2f}%")
print(f"鸢尾花数据集的降维后维数为 {clf.projectMatrix.shape[1]}")
