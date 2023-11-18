import numpy as np
import matplotlib.pyplot as plt


class PCA:
    """ 主成分分析类，实现维数约简 """
    """ PCA 类的使用说明
        1. pca = PCA(data, threshold=0.85, dimension=None)  
            创建 PCA 对象：传入高维数据，指定重构阈值[可选]，指定主成分数量[可选]
        2. projectedData = pca.project() 
            返回高维数据的低维投影
        3. pca.visualize()
            可视化高维数据的低维投影
    """

    def __init__(self, data, threshold=0.85, dimension=None):
        self.data = data  # 原始数据矩阵，n × m，样本行向量，共计 n 个样本，每个样本 m 维特征
        self.threshold = threshold  # 重构阈值（or 最小累计解释方差），用于选择主成分的数量（低维空间的维数）
        self.dimension = dimension  # 指定主成分的数量，if None，通过默认的重构阈值选择主成分，else 忽略重构阈值

        self._projectMatrix = None  # 投影矩阵，PCA 的目标
        self._cov = None  # 高维数据样本中心化处理之后的协方差矩阵
        self._sorted_eigenvalues = None  # 协方差矩阵的升序排列的特征值
        self._sorted_eigenvectors = None  # 协方差矩阵升序排列的特征值对应的特征向量
        self._cev = None  # 协方差矩阵的前 i 个特征值对应的累计解释方差

    def _fit(self):
        """ 核心函数，根据传入数据得到投影矩阵 """
        # 0.数据预处理（数据标准化）
        # 标准化：即一列的每个元素，减去该列的均值后，除以该列的方差
        data = np.array(self.data, dtype=float)  # 确保数据矩阵是 ndarray 类型，其中数据是浮点数
        mean = np.mean(data, axis=0)  # 样本均值，行向量
        std_dev = np.std(data, axis=0)  # 样本标准差，行向量
        normalized_data = ((data - mean) / std_dev).T  # 标准化后的数据矩阵，且使一个样本对应一个列向量

        # 1.样本中心化：即将每列的数据减去每列均值，数据预处理阶段已经完成该内容

        # 2.计算协方差矩阵 XX^T (shape=m×m)
        cov = np.dot(normalized_data, normalized_data.T)
        self._cov = cov

        # 3.对协方差矩阵做特征值分解
        eigenvalues, eigenvectors = np.linalg.eig(cov)  # 特征值和对应的特征向量（列向量）

        # 4.选择主成分
        # 基于特征值对特征值和特征向量降序排列
        sort_indices = np.argsort(eigenvalues)[::-1]  # 获取特征值降序排序索引
        sorted_eigenvalues = eigenvalues[sort_indices]  # 使用索引值对特征值和特征向量进行排序
        sorted_eigenvectors = eigenvectors[:, sort_indices]
        self._sorted_eigenvalues = sorted_eigenvalues
        self._sorted_eigenvectors = sorted_eigenvectors

        # 基于重构阈值或者指定维数，选择主成分
        if not self.dimension:  # 未指定维数，则基于重构阈值选择主成分
            # 计算累计解释方差，cev[i] 表示前 i 个特征值的累计解释方差
            cev = []
            for i in range(len(sorted_eigenvalues)):
                cev_i = sum(sorted_eigenvalues[:i + 1]) / sum(sorted_eigenvalues)
                cev.append(cev_i)
            self._cev = cev
            # 基于重构阈值，选择主成分
            for i in range(len(cev)):
                if cev[i] >= self.threshold:  # 满足重构阈值，则确定投影矩阵
                    self._projectMatrix = sorted_eigenvectors[:, :i + 1]
                    break
        else:  # 基于指定维数选择主成分
            if self.dimension >= normalized_data.shape[0]:  # 指定的维数不能大于等于原样本维数
                raise Exception("function fit: illegal dimension")
            self._projectMatrix = sorted_eigenvectors[:, :self.dimension]

        # 更新参数：主成分数量
        self.dimension = self._projectMatrix.shape[1]

        # 5.数据投影：将原始数据投影到主成分空间中
        # 需调用 PCA 的方法，project

    def project(self):
        """ 获取投影后的数据（低维数据） """
        # self.data n 行样本，m 维特征；self.projectMatrix m 维载荷因子，k 个主成分（PCA）
        self._fit()

        projectedData = np.dot(self.data, self._projectMatrix)

        return projectedData

    def visualize(self, target, dimension=2):
        """ 对降维后的数据选择两个主成分 or 三个主成分进行可视化操作 """
        # target 是 self.data 每行对应的样本的类别信息，基于此可以对相同的样本点以相同的颜色表示
        target = np.array(target, dtype=float)

        # 获取样本的类别情况
        unique_labels = np.unique(target)

        # 将降维后的数据和类别标签拼接
        target = target.reshape(-1, 1)
        data = np.concatenate((self.project(), target), axis=1)

        if dimension > self.dimension:
            raise Exception("function visualize: illegal dimension")

        # 替换为系统中存在的中文字体
        plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'

        if dimension == 2:  # 二维可视化
            # 可视化
            for label in unique_labels:
                class_points = data[data[:, -1] == label]  # 获取类别为 label 的样本点
                plt.scatter(class_points[:, 0], class_points[:, 1], label=f"Class {label}")

            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title("主成分分析可视化散点图")
            plt.legend()
            plt.show()
        elif dimension == 3:  # 三维可视化
            figure = plt.figure()
            ax = figure.add_subplot(111, projection="3d")

            # 可视化
            for label in unique_labels:
                class_points = data[data[:, -1] == label]  # 获取类别为 label 的样本点
                print(class_points)
                ax.scatter(class_points[:, 0], class_points[:, 1], class_points[:, 2], label=f'Class {label}')

            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")
            ax.set_title("主成分分析可视化散点图")
            ax.legend()
            plt.show()


if __name__ == "__main__":
    # 载入数据
    from sklearn.datasets import load_iris

    iris_dataset = load_iris()
    iris_target = iris_dataset['target']  # 数据标签
    iris_target_names = iris_dataset['target_names']  # 数据标签名
    iris_data = iris_dataset['data']  # 数据集
    iris_feature_names = iris_dataset['feature_names']  # 数据集特征名

    # 主成分分析
    pca = PCA(iris_data)

    # 可视化主成分分析后的结果
    pca.visualize(iris_target, 2)  # 二维可视化
    # pca.visualize(iris_target, 3)  # 三维可视化
    # print(pca.dimension)
