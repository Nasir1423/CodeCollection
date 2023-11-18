import numpy as np


class LogisticRegression_newton():
    """ 对数几率回归模型 Y=w^T*X+b=beta^T*X_hat """

    def __init__(self, learning_rate=0.5, max_iterations=10):
        """
        :param learning_rate: float, 学习率
        :param max_iterations: int, 最大迭代次数
        """
        self.beta = None
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def initialize_weights(self, m_features):
        """
        :param m_features: 整型，特征维数
        :return: 该方法用于初始化及调整权重
        """
        # 方法调用后，self.beta=[[w_1],[w_2],[w_3],...,[w_m],[0]] (列向量, (m+1,1))
        limit = np.sqrt(1 / m_features)
        w = np.random.uniform(-limit, limit, (m_features, 1))
        b = np.asarray([[0]])
        self.beta = np.append(w, b, axis=0)  # beta=[w;b]，待求参数

    def fit(self, X, y):
        """
        :param X: 2 dimensions matrix 样本数据集，矩阵的每一行是一个样本，即一个行向量是一个样本
        :param y: column vector 样本标签
        :return: 该方法用于求解模型参数 beta
        """
        n_samples, m_features = X.shape
        self.initialize_weights(m_features)

        """ 给 X 的每个样本插入数字 1 (为了凑出 Y=beta^T*X_hat 的形式);
            将 y 的形状变为列向量的形式，便于后续计算
            X = [[x_11,x_12,x_13,...,x_1m],
                 [x_21,x_22,x_23,...,x_2m],
                 [x_31,x_32,x_33,...,x_3m],
                 ...
                 [x_n1,x_n2,x_n3,...,x_nm]] (n, m)
            y = [[y_1],[y_2],[y_3],...,[y_n]] (列向量, (n, 1))
            X_hat =[[x_11,x_12,x_13,...,x_1m,1],
                    [x_21,x_22,x_23,...,x_2m,1],
                    [x_31,x_32,x_33,...,x_3m,1],
                    ...
                    [x_n1,x_n2,x_n3,...,x_nm,1]] (n, m+1)
        """
        X_plus_one = np.ones((n_samples, 1))
        X = np.append(X, X_plus_one, axis=1)  # X = X_hat=[X;X_plus], shape of X is (n, m+1)
        y = np.reshape(y, (n_samples, 1))  # shape of y is (n,1)

        """ 模型训练(核心): 基于 X_hat、y、beta 进行训练
            repeat
            1. 计算牛顿步长和牛顿减量
            2. 停止判断
            3. 回溯直线搜索，计算学习率 t
            4. 更新参数 β
        """
        for i in range(self.max_iterations):
            tolerance = 10 ** -6  # 设定停止阈值
            # step1. 计算牛顿步长和牛顿减量
            der_first = np.zeros((1, m_features + 1))  # 梯度，或者说是损失函数关于参数 beta 的一阶导数
            der_second = np.zeros((m_features + 1, m_features + 1))  # 海森矩阵，或者说是损失函数关于参数 beta 的二阶导数
            for j in range(n_samples):
                x = X[j, :].reshape(1, -1)  # 取矩阵的第 j 行，并转换为行向量
                eta = np.dot(x, self.beta)
                p1 = np.exp(eta) / (1 + np.exp(eta)) if eta <= 0 else 1.0 / (1 + np.exp(-eta))
                p0 = 1 - p1
                der_first -= x * (y[j] - p1)
                der_second += np.dot(x.T, x) * p1 * p0  # der_second 是矩阵 (m+1, m+1)
            der_first = der_first.reshape(-1, 1)  # der_first 是列向量 (m+1, 1)
            step_size_newton = -np.dot(np.linalg.pinv(der_second), der_first)  # 牛顿步长，是一个列向量，(m+1, 1)
            decrement_newton = np.dot(der_first.T, -step_size_newton)  # 牛顿减量，是一个数字
            # step2. 停止判断
            if decrement_newton / 2 <= tolerance:
                break
            # step3. 回溯直线搜索 backtracking line search
            alpha_search = 0.3
            beta_search = 0.5
            self.learning_rate = 1.0
            while True:
                """ 优化函数为 min: l(self.beta)=SUM[-y_i*self.beta^T*x+ln(1+e^{self.beta^Tx})] """
                fun_plus = 0  # 分别计算两个优化函数值
                fun = 0
                beta_plus = self.beta + self.learning_rate * step_size_newton  # 列向量
                for j in range(n_samples):
                    x = X[j, :].reshape(-1, 1)  # 取出 x 为列向量
                    fun_plus += -y[j] * np.dot(beta_plus.T, x) + np.log(1 + self.sigmod(x.T, beta_plus))
                    fun = -y[j] * np.dot(self.beta.T, x) + np.log(1 + self.sigmod(x.T, self.beta))
                if fun_plus <= fun - alpha_search * self.learning_rate * decrement_newton:
                    break
                self.learning_rate = beta_search * self.learning_rate
            # step4. 参数更新
            self.beta += self.learning_rate * step_size_newton

    def predict(self, X_test):
        """
        :param X_test: 测试数据集，矩阵的每一行是一个样本，即一个行向量是一个样本
        :return: 列向量，表示测试数据集的预测值 (0 还是 1)
        """
        n_test_samples = X_test.shape[0]
        y_predict = np.zeros((n_test_samples, 1))  # 预测值列向量

        for i in range(n_test_samples):
            x = X_test.iloc[i, :].values  # 测试样本行向量
            x = np.asarray(x).reshape(1, -1)
            x = np.append(x, np.ones((1, 1)), axis=1)
            y_predict[i] = 0 if self.sigmod(x, self.beta) <= 0.5 else 1

        return y_predict

    def accuracy(self, X_test, y_test):
        """
        :param X_test: 测试数据集，矩阵的每一行是一个样本，即一个行向量是一个样本
        :param y_test: 预测值列向量
        :return: 返回模型预测准确度
        """
        n_test_samples = X_test.shape[0]
        y_predict = self.predict(X_test)
        hit_samples = 0  # 表示模型预测准确的样本量

        for i in range(n_test_samples):
            if y_test[i] == y_predict[i]:
                hit_samples += 1
        accuracy = 1.0 * hit_samples / n_test_samples
        return accuracy

    @staticmethod
    def sigmod(x, beta):
        """
        :param beta: (m+1, 1) 列向量，beta=[w;b]=[[w_1],[w_2],[w_3],...,[w_m],[0]]
        :param x: (1, m+1) 一个样本数据，行向量，x=[[x_1,x_2,x_3,...,x_m,1]]
        :return: p in (0,1) 对数几率函数，返回对结果的可能性，大于 0.5 是正例，小于 0.5 是反例
        """
        eta = np.dot(x, beta)  # a real number
        if eta >= 0:  # 引入 if-else 结构避免计算概率值时，exp() 的值过大移出 (避免上溢)
            possibility = 1.0 / (1.0 + np.exp(-eta))
        else:
            possibility = np.exp(eta) / (1.0 + np.exp(eta))

        return possibility


"""
    注意事项
    1. [1,2,3] 不是行向量，也不是列向量
       [[1],[2],[3]] 是列向量，[[1,2,3]] 是行向量
       其中
        np.asarray([1,2,3]).reshape(-1,1) 得到列向量
        np.asarray([1,2,3]).reshape(1,-1) 得到行向量
    2. 所有的运算一定要转换为向量的形式，避免出错
    3. X 的每一行是一个样本；x 是行向量；beta 是列向量；y 是列向量
    4. 模型输入的参数最好是 DataFrame 类型的，因为以上代码使用了许多相关函数
"""
