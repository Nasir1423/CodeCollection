import numpy as np
import pandas as pd

from LoanApprovalPredict.model.DecisionTree.BinaryTree import *


class DecisionTree:
    """ 决策树模型，用于二分类 """

    def __init__(self, min_samples_split=5, max_depth=6, criterion="entropy"):
        self.min_samples_split = min_samples_split  # 结点可分最小样本量(小于则直接划分为叶结点)
        self.max_depth = max_depth  # 决策树的最大深度(根节点是零层)
        self.criterion = criterion  # 最优化分属性的判断依据
        # 一般而言，有信息增益 entropy, 增益率 entropy_ratio, 基尼系数 gini 三种方法，本决策树类仅实现了第一种方法
        self.tree = None  # 创建一个空二叉树作为决策树

    # 训练决策树
    def fit(self, data):
        if not self.tree:  # 决策树为空
            self.tree = self.create_tree(data)

    # 递归生成决策树
    def create_tree(self, data, depth=0):
        newNode = TreeNode()  # 新建结点
        data_label = data.iloc[:, -1]  # 数据集标签列

        # 1.递归返回判断
        # 当前结点包含的样本全部属于同一类别，则无需划分
        if len(data_label.value_counts()) == 1:
            newNode.node_class = data_label.values[0]  # 结点类别
            newNode.node_type = NodeStatus.LEAF  # 结点类型(叶结点)
            newNode.samples = data.shape[0]  # 结点对应的样本数量
            newNode.depth = depth
            return BinaryTree(newNode)
        # 当前结点属性集为空(只剩数据标签列) or 所有样本在所有属性上取值相同 or 当前结点的属性数量少于 5
        if data.shape[1] == 1 or \
                all([len(data[fea].value_counts()) == 1 for fea in data.columns[:-1]]) or \
                data.shape[0] < 5 or \
                depth == self.max_depth:
            newNode.node_class = DecisionTree.get_most_label(data)
            newNode.node_type = NodeStatus.LEAF  # 结点类型(叶结点)
            newNode.samples = data.shape[0]  # 结点对应的样本数量
            newNode.depth = depth
            return BinaryTree(newNode)

        # 2.递归生成决策树
        best_feature_info = DecisionTree.get_best_feature(data)  # 最优化分属性相关信息(连续和离散属性返回的变量数目不同)
        if len(best_feature_info) == 3:  # 连续属性
            best_feature_type = FeatureStatus.CONTINUOUS
        else:  # 离散属性
            best_feature_type = FeatureStatus.DISCRETE

        if best_feature_type == FeatureStatus.CONTINUOUS:  # 对于连续属性
            best_feature, information_gain, threshold = best_feature_info
        else:  # 对于离散属性
            best_feature, information_gain = best_feature_info
            threshold = data[best_feature].value_counts().keys()[0]

        # 填充结点信息
        newNode.feature_name = best_feature
        newNode.feature_type = best_feature_type
        newNode.threshold = threshold
        newNode.entropy = information_gain
        newNode.samples = data.shape[0]
        newNode.node_type = NodeStatus.NON_LEAF
        newNode.node_class = DecisionTree.get_most_label(data)
        newNode.depth = depth

        # 生成树对象(待插入左右子树)
        newTree = BinaryTree(newNode)

        # 生成左右子树
        if best_feature_type == FeatureStatus.CONTINUOUS:  # 对于连续属性
            data_t_less = data.loc[data[best_feature] <= threshold].drop(best_feature, axis=1)
            data_t_greater = data.loc[data[best_feature] > threshold].drop(best_feature, axis=1)
            if not data_t_less.empty:  # 添加左子树
                child = self.create_tree(data_t_less, depth=depth + 1)
                newTree.insert_left(child)
            if not data_t_greater.empty:  # 添加右子树
                child = self.create_tree(data_t_greater, depth=depth + 1)
                newTree.insert_right(child)
        else:  # 对于离散属性
            data_equals = data.loc[data[best_feature] == threshold].drop(best_feature, axis=1)
            data_unequals = data.loc[data[best_feature] != threshold].drop(best_feature, axis=1)
            if not data_equals.empty:  # 添加左子树
                child = self.create_tree(data_equals, depth=depth + 1)
                newTree.insert_left(child)
            if not data_unequals.empty:  # 添加右子树
                child = self.create_tree(data_unequals, depth=depth + 1)
                newTree.insert_right(child)

        # 返回生成的树对象
        return newTree

    # 返回模型预测数据集的准确度
    def accuracy(self, data):
        data_label = data.iloc[:, -1]
        data_predict = self.predict(data)

        count = 0
        data_scale = len(data_label)

        for i in range(data_scale):
            if data_label.iloc[i] == data_predict.iloc[i]:
                count += 1

        acc = count / data_scale
        return acc

    # 预测数据分类
    def predict(self, data):
        samples = data.iloc[:, :-1]  # 将数据集去除标签列
        result = []  # 存储预测结果

        for i in range(data.shape[0]):
            sample = samples.iloc[i, :]

            current_tree = self.tree  # 从根结点进行判断
            current_node_type = current_tree.node.node_type  # 当前结点的状态(是不是叶子节点)
            current_feature_type = current_tree.node.feature_type  # 当前属性的状态(离散还是连续)

            while current_node_type == NodeStatus.NON_LEAF:  # 只要没有遍历到叶结点就一直遍历
                if current_feature_type == FeatureStatus.DISCRETE:  # 离散属性
                    if current_tree.node.threshold == sample[current_tree.node.feature_name]:  # 遍历到左子树
                        current_tree = current_tree.left_child
                    else:  # 遍历到右子树
                        current_tree = current_tree.right_child
                else:  # 连续属性
                    if current_tree.node.threshold >= sample[current_tree.node.feature_name]:
                        current_tree = current_tree.left_child
                    else:
                        current_tree = current_tree.right_child

                current_node_type = current_tree.node.node_type
                current_feature_type = current_tree.node.feature_type

            result.append(current_tree.node.node_class)

        return pd.Series(result)

    # 验证模型准确度

    # # 利用决策树模型对测试集数据进行测试，返回准确度
    # def accuracy(self, data):
    #     predict_list = pd.Series()
    #     hit_num = 0
    #     label = data.iloc[:, -1]
    #     data_without_label = data.iloc[:, :-1]
    #     for i in range(data.shape[0]):
    #         sample = data_without_label.iloc[i]
    #         sample_label = label.iloc[i]
    #         pre_res = DecisionTree_entropy.predict(self.tree, sample)
    #         if sample_label == pre_res:
    #             hit_num += 1
    #     return hit_num / data.shape[0]
    #
    # 计算数据集的信息熵
    @staticmethod
    def cal_information_entropy(data):
        data_label = data.iloc[:, -1]  # 数据集标签列
        label_class = data_label.value_counts()  # 标签频次统计
        Ent = 0  # 数据集的信息熵
        for k in label_class.keys():
            p_k = label_class[k] / len(data_label)
            Ent += -p_k * np.log2(p_k)
        return Ent

    # 计算数据集以属性 a 划分的信息增益
    # 如果是离散属性，返回信息增益; 如果是连续属性，返回信息增益，最优化分点
    @staticmethod
    def cal_information_gain(data, a):
        Ent = DecisionTree.cal_information_entropy(data)  # 数据集信息熵
        feature_class = data[a].value_counts()  # 属性为 a 的数据列进行频次统计

        # 判断属性 a 是连续还是离散
        if len(feature_class.keys()) < 3:  # 离散(属性为 a 的数据列只有两种取值，或更少)
            feature_type = FeatureStatus.DISCRETE
        else:
            feature_type = FeatureStatus.CONTINUOUS

        # 根据属性类型求解信息增益
        if feature_type == FeatureStatus.DISCRETE:  # 离散属性
            gain = 0
            for k in feature_class.keys():
                weight = feature_class[k] / data.shape[0]
                data_v = data.loc[data[a] == k]  # data 在属性 a 上取值为 v 的样本集 (.loc 根据 data[a]==v 的真值 Series 对象，选择符合条件的样本)
                Ent_v = DecisionTree.cal_information_entropy(data_v)
                gain += weight * Ent_v
            return [Ent - gain]
        else:  # 连续属性
            # 将属性为 a 的数据列取出、升序排序、去重、重置索引
            sorted_feature = data[a].sort_values().drop_duplicates().reset_index(drop=True)
            # 1. 获取候选划分点集合
            T = []  # 候选划分点集合
            for i in range(len(sorted_feature) - 1):
                t = (sorted_feature[i] + sorted_feature[i + 1]) / 2
                T.append(t)
            # 2. 基于信息增益准则选取最优的划分点
            # 获取不同候选划分点下的样本集合的信息增益
            T_gain = []  # 不同候选划分点下的样本集合的信息增益
            for t in T:
                # 以候选划分点 t 划分数据集 data
                data_t_less = data.loc[data[a] <= t]
                data_t_greater = data.loc[data[a] > t]
                # 计算划分后的两个数据集的信息熵
                Ent_t_less = DecisionTree.cal_information_entropy(data_t_less)
                Ent_t_greater = DecisionTree.cal_information_entropy(data_t_greater)
                # 计算划分后的两个数据集的权重
                weight_t_less = np.sum(data[a] <= t) / data.shape[0]
                weight_t_greater = np.sum(data[a] > t) / data.shape[0]
                # 计算信息增益
                gain_a_t = Ent - weight_t_less * Ent_t_less - weight_t_greater * Ent_t_greater
                T_gain.append(gain_a_t)
            # 选择最优候选划分点
            best_t_gain = max(T_gain)  # 最大信息增益
            threshold = T[T_gain.index(best_t_gain)]  # 最优划分点
            return [best_t_gain, threshold]

    # 返回数据集中大多数样本所属的类
    @staticmethod
    def get_most_label(data):
        data_label = data.iloc[:, -1]
        label_sort = data_label.value_counts(sort=True)
        most_label = label_sort.keys()[0]
        return most_label

    # 获取最优划分属性
    # 如果是离散属性，返回最优化分属性，对应信息增益；如果是连续属性，返回最优化分属性，对应信息增益，对应划分点
    @staticmethod
    def get_best_feature(data):
        features = data.columns[0:-1]  # 所有属性名
        res = {}  # 字典存储信息增益，key 是属性名，value 是对应的信息增益
        for a in features:
            temp = DecisionTree.cal_information_gain(data, a)[0]  # 计算不同属性的信息增益
            res[a] = temp
        res = res.items()  # 以列表的形式返回可遍历的元组数组，每个元组由一个键值对组成(属性:增益)
        res = sorted(res, key=lambda x: x[1], reverse=True)  # 基于信息增益进行降序排序

        best_feature = res[0][0]  # 最优划分属性

        # 判断 best_feature 是连续还是离散
        feature_class = data[best_feature].value_counts()  # 属性为 a 的数据列进行频次统计
        if len(feature_class.keys()) < 3:  # 离散(属性为 a 的数据列只有两种取值，或更少)
            feature_type = FeatureStatus.DISCRETE
        else:
            feature_type = FeatureStatus.CONTINUOUS

        if feature_type == FeatureStatus.DISCRETE:  # 离散属性
            best_information_gain = res[0][1]
            return [best_feature, best_information_gain]
        else:  # 连续属性
            best_information_gain, threshold = DecisionTree.cal_information_gain(data, best_feature)
            return [best_feature, best_information_gain, threshold]

    """
        注意
        1. data 的格式为
        [[x_11,x_12,...,x_1m,y_1],
         [x_21,x_22,...,x_2m,y_2],
         ...
         [x_n1,x_n2,...,x_nm,y_n],]
    """
