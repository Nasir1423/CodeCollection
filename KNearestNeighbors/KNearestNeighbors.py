import numpy as np


class KNearestNeighbors:

    def __init__(self, X_train, y_train, n_neighbors=3):
        self.X_train = X_train
        self.y_train = y_train
        self.n_neighbors = n_neighbors

    def predict(self, X_test):
        # 暴力求解，速度慢
        predictions = []

        for sample in X_test:
            distances = np.sqrt(np.sum((self.X_train - sample) ** 2, axis=1))  # 计算当前样本到已知的每个样本的距离
            indices = np.argsort(distances)[:self.n_neighbors]  # 获得 K 近邻的样本数据索引
            knn_labels = self.y_train[indices]  # 获得 K 近邻的样本数据标签
            most_common = np.bincount(knn_labels).argmax()
            predictions.append(most_common)

        return np.asarray(predictions)

    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        count_same_samples = np.sum(predictions == y_test)
        score = count_same_samples * 1.0 / y_test.shape[0]
        return score
