import numpy as np
import matplotlib.pyplot as plt


class LinearDiscriminantAnalysis:
    """ 线性判别分析: 可实现数据降维或分类 """
    """
        clf = LinearDiscriminantAnalysis(X, y)
        clf.fit()
        clf.project()
        clf.visualize()
        clf.predict(X_new)
        def.score(X_new, y_new)
    """

    def __init__(self, X, y):
        self.X = X  # 特征矩阵 n × m (n 行样本，每个样本包含 m 维特征)；注意，传入的数据是一个行向量对应一个样本
        self.y = y  # 目标类别 n × 1

        self.projectMatrix = None  # 投影矩阵 (d' × m)；投影数据 (d' × n)，原始数据 (m × n)；数据一般投影到 “类别-1” 维
        self.classifiers = None  # 分类器列表，包含多个分类器，用于分类；为了兼容多分类问题，c 分类对应 c 个分类器

    class classifier:
        def __init__(self, X, y, label):
            self.w, self.w0 = self.init_weight_threshold(X, y, label)

        def score(self, x):
            # 返回样本 x 对应当前分类器类别的概率
            score = np.dot(self.w.T, x) - self.w0

            return score if score > 0 else 0

        @staticmethod
        def init_weight_threshold(X, y, label):
            # 根据传入的数据，及类别标签初始化当前分类器的权重和阈值参，label 是当前分类器的正例类别
            # 返回 w 和 w0

            # 选取指定类别的样本
            positive_samples = X[y == label]  # 正例（对应 label）
            negative_samples = X[y != label]  # 负例

            # 计算均值向量
            mean_positive = np.mean(positive_samples, axis=0)
            mean_negative = np.mean(negative_samples, axis=0)

            # 计算类内离散度矩阵
            S1 = np.dot((positive_samples - mean_positive).T, (positive_samples - mean_positive))
            S2 = np.dot((negative_samples - mean_negative).T, (negative_samples - mean_negative))
            Sw = S1 + S2

            # 解投影权值
            w = np.dot(np.linalg.inv(Sw), (mean_positive - mean_negative).T)

            # 解分类阈值
            w0 = (np.dot(w.T, mean_positive.T) + np.dot(w.T, mean_negative.T)) / 2

            return w, w0

    def fit(self):
        """ 计算投影矩阵及分类器 """

        # 第一部分 计算投影矩阵
        # 1.计算类内离散度矩阵和类间离散度矩阵
        # 1.0 获取样本类别标签
        class_labels = np.unique(self.y)
        # 1.1 计算所有样本的均值
        mean_overall = np.mean(self.X, axis=0)
        # 1.2 初始化类内和类间的离散度矩阵，m × m
        S_W = np.zeros((self.X.shape[1], self.X.shape[1]))
        S_B = np.zeros((self.X.shape[1], self.X.shape[1]))
        # 1.3 计算类内和类间离散度矩阵
        for label in class_labels:
            class_samples = self.X[self.y == label]  # 获取当前类别的样本
            mean_class = np.mean(class_samples, axis=0)  # 获取当前类别的样本均值
            class_number = len(class_samples)  # 获取当前类别的样本数量
            S_W += np.dot((class_samples - mean_class).T, (class_samples - mean_class))  # 计算当前类别的类内离散度矩阵
            S_B += class_number * np.dot((mean_class - mean_overall), (mean_class - mean_overall).T)  # 计算类间离散度矩阵的一部分

        # 2.特征分解
        eigenvalues, eigenvectors = np.linalg.eig(np.dot(np.linalg.inv(S_W), S_B))

        # 3.获取投影矩阵(前 d' 个特征向量)
        sorted_indices = np.argsort(eigenvalues)[::-1]  # 降序排序索引
        # sorted_eigenvalues = eigenvalues[sorted_indices]  # 降序排列特征值
        sorted_eigenvectors = eigenvectors[:, sorted_indices]  # 降序排列特征向量

        self.projectMatrix = sorted_eigenvectors[:, : len(class_labels) - 1]  # 获得投影矩阵

        # 第二部分 计算 ovr 多类分类器
        classifiers = {}
        for label in class_labels:
            classifiers[label] = self.classifier(self.X, self.y, label)
        self.classifiers = classifiers

    def project(self, X_new=None):
        """ 获取投影后的数据: 如果没有传入 data 参数，则返回用于训练的样本的投影后数据 """
        # 注意 X 和 projectedX 都是一行代表一个样本
        X = X_new if X_new else self.X
        projectedX = np.dot(X, self.projectMatrix)
        return projectedX

    def visualize(self, X_new, y_new):
        """ 数据可视化（所有数据都可视化到二维情况） """

        if self.projectMatrix.shape[1] > 2:
            projectedX = np.dot(X_new, self.projectMatrix[:, :2])
            plt.scatter(projectedX[:, 0], projectedX[:, 1], marker='o', c=y_new)
        else:
            projectedX = np.dot(X_new, self.projectMatrix[:, :1])
            plt.scatter(projectedX[:, 0], np.zeros(projectedX[:, 0].shape), marker='o', c=y_new)

        plt.title("LDA")
        plt.show()

    def predict(self, X_new):
        """ 返回对 X_new 的预测结果列向量 """
        predict_res = []  # 预测结果列表

        for x in X_new:  # 遍历每个新的样本
            x_score = {}  # 当前样本不同类别及对应的评分
            for label, classifier in self.classifiers.items():
                x_score[label] = classifier.score(x)
            max_possible_key = max(x_score, key=x_score.get)
            predict_res.append(max_possible_key)

        return np.asarray(predict_res).reshape(-1, 1)

    def score(self, X_new, y_new):
        """ 返回预测准确率 """
        predict_res = self.predict(X_new)
        y_new = y_new.reshape(-1, 1)
        score = np.count_nonzero(predict_res == y_new) / len(y_new)
        return score
