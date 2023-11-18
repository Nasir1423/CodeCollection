"""
    工具文件，写了一些工具函数
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn import preprocessing
from scipy.interpolate import make_interp_spline
from LoanApprovalPredict.model.LogisticRegression.LogisticRegression_newton import *
from LoanApprovalPredict.model.DecisionTree.DecisionTree import *

""" loan_approval_data: 4296 rows × 13 columns, 且数据集干净，无需缺失值处理等
    loan_id                      int64  贷款批准样本的 id (无用特征)
    no_of_dependents             int64  feature1: Number of Dependents of the Applicant(家属个数), int
    education                   object  feature2: 受教育情况(Graduate/Not Graduate), str; Mapping(Graduate:1, Not Graduate:0)
    self_employed               object  feature3: 是否为自雇人士(Yes/No), str; Mapping(Yes:1, No:0)
    income_annum                 int64  feature4: 年收入, int
    loan_amount                  int64  feature5: 贷款数额, int
    loan_term                    int64  feature6: 贷款期限, int
    cibil_score                  int64  feature7: 信用分数, int
    residential_assets_value     int64  feature8: 住宅资产价值, int
    commercial_assets_value      int64  feature9: 商业资产价值, int
    luxury_assets_value          int64  feature10: 奢侈品资产价值, int
    bank_asset_value             int64  feature11: 银行资产价值, int
    loan_status                 object  y: 是否借贷(Approval/Rejected), str; Mapping(Approval:1, Rejected:0)
    综上所述，一个样本总共有 11 个属性 or 特征，1 个值
"""


# 导入、预处理、[划分]数据
def loan_approval_data_processed(directory='../data/loan_approval_dataset.csv', minmax=True, split=True, scale=0.75):
    """
    :param directory: loan_approval_predict 数据集位置
    :param minmax: 是否进行归一化
    :param split: 是否将数据集划分为训练集和测试集
    :param scale: 训练集比例
    :return: train, test
    """
    # 1. 导入数据
    loan_approval_data = pd.read_csv(directory)

    # 2. 数据预处理(由于提供的数据比较干净：无缺失值，因此只需要进行①删除指定列②将字符串映射为值，这两项工作即可)
    loan_approval_data.drop("loan_id", axis=1, inplace=True)  # 删除 loan_id 这一列的数据
    loan_approval_data.columns = loan_approval_data.columns.str.strip()  # 清除列名的前后空格
    # loan_approval_data.columns = [name.strip() for name in loan_approval_data.columns]  # 清除列名的前后空格
    # 2.1 将属性 education 中的 Graduated 映射为 1，Not Graduated 映射为 0
    loan_approval_data.loc[:, "education"] = loan_approval_data.loc[:,
                                             "education"].str.strip()  # 清除列 "education" 每一项的前后空格
    loan_approval_data.loc[:, "education"] = loan_approval_data.loc[:, "education"].map(
        {'Graduate': 1, 'Not Graduate': 0}).astype(int)
    loan_approval_data["education"] = pd.to_numeric(loan_approval_data["education"],
                                                    errors='coerce')  # 更改该列数据的属性为 int 类型
    # 2.2 将属性 self_employed 中的 Yes 映射为 1，No 映射为 0
    loan_approval_data.loc[:, "self_employed"] = loan_approval_data.loc[:,
                                                 "self_employed"].str.strip()  # 清除列 "self_employed" 每一项的前后空格
    loan_approval_data.loc[:, "self_employed"] = loan_approval_data.loc[:, "self_employed"].map(
        {'Yes': 1, 'No': 0}).astype(int)
    loan_approval_data["self_employed"] = pd.to_numeric(loan_approval_data["self_employed"],
                                                        errors='coerce')  # 更改该列数据的属性为
    # 2.3 将属性 loan_status 中的 Approval 映射为 1，Rejected 映射为 0
    loan_approval_data.loc[:, "loan_status"] = loan_approval_data.loc[:,
                                               "loan_status"].str.strip()  # 清除列 "loan_status" 每一项的前后空格
    loan_approval_data.loc[:, "loan_status"] = loan_approval_data.loc[:, "loan_status"].map(
        {'Approved': 1, 'Rejected': 0}).astype(int)
    loan_approval_data["loan_status"] = pd.to_numeric(loan_approval_data["loan_status"], errors='coerce')  # 更改该列数据的属性为
    # 2.4 对数据进行归一化操作
    processed_data = loan_approval_data
    if minmax:
        minmax_scaler = preprocessing.MinMaxScaler()
        minmax_data = minmax_scaler.fit_transform(loan_approval_data)
        processed_data = pd.DataFrame(minmax_data, columns=loan_approval_data.columns)

    # 3. 划分训练集和验证集
    if split:
        train = processed_data.sample(frac=scale, random_state=int(time.time()))
        test = processed_data.drop(train.index)
        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)

        return train, test
    else:
        return processed_data


# 将数据集划分为样本数据、标签
def dataset_split(data):
    X = data.iloc[:, 0:-1]
    y = data.iloc[:, -1]
    return X, y


def draw_learning_curve(data, model, start=0.001, end=0.3, step=0.001):
    plt.xlabel("Number of training samples")  # 横轴
    plt.ylabel("Accuracy")  # 纵轴

    # 计算出对应不同训练集规模的时，模型在训练集和测试集合上的比重
    train_size_list = list(np.arange(start, end, step))  # 训练集占全部数据集的比重
    number_of_training_samples_list = [int(size * data.shape[0]) for size in
                                       train_size_list]  # 训练集的样本数量(横坐标的值)
    train_accuracy_list = []  # 训练集随样本数的准确度取值(纵坐标)
    test_accuracy_list = []  # 测试集随样本数的准确度取值(纵坐标)
    # 获取两个图像的纵坐标取值
    for frac in train_size_list:
        # a = time.time()
        # 按照预定的比例划分数据集和测试集合
        train = data.sample(frac=frac, random_state=int(time.time()))
        test = data.drop(train.index)

        if isinstance(model, LogisticRegression_newton):
            plt.title("learning curve(Logistic Regression)")  # 标题
            # 重置索引
            train.reset_index(drop=True, inplace=True)
            test.reset_index(drop=True, inplace=True)
            # 划分训练集和测试集的样本数据、标签
            X_train, y_train = dataset_split(train)
            X_test, y_test = dataset_split(test)
            # 模型训练
            model.fit(X_train, y_train)
            # 添加纵坐标值
            train_accuracy = model.accuracy(X_train, y_train)
            train_accuracy_list.append(train_accuracy)
            test_accuracy = model.accuracy(X_test, y_test)
            test_accuracy_list.append(test_accuracy)
        else:
            plt.title("learning curve(Decision Tree)")  # 标题
            # 重置索引
            train.reset_index(drop=True, inplace=True)
            test.reset_index(drop=True, inplace=True)
            # 模型训练
            model.fit(train)
            # 添加纵坐标值
            train_accuracy = model.accuracy(train)
            train_accuracy_list.append(train_accuracy)
            test_accuracy = model.accuracy(test)
            test_accuracy_list.append(test_accuracy)

        # b = time.time()
        # print("\nfrac=%.2f" % frac)
        # print("time=%.2f" % (b - a))
        # print("train_acc%.2f" % train_accuracy)
        # print("test_acc%.2f" % test_accuracy)

    # 绘图
    # 对 x、y_train 插值
    number_of_training_samples_list_smooth = np.linspace(min(number_of_training_samples_list),
                                                         max(number_of_training_samples_list), 50)
    train_accuracy_list_smooth = make_interp_spline(number_of_training_samples_list, train_accuracy_list)(
        number_of_training_samples_list_smooth)
    plt.plot(number_of_training_samples_list_smooth, train_accuracy_list_smooth, color='r', label='Training Score')
    # 对 y_test 插值
    test_accuracy_list_smooth = make_interp_spline(number_of_training_samples_list, test_accuracy_list)(
        number_of_training_samples_list_smooth)
    plt.plot(number_of_training_samples_list_smooth, test_accuracy_list_smooth, color='g', label='Test Score')
    plt.legend(loc='best')
    plt.show()


# 返回模型训练时间
def time_consumption(data, model):
    start = time.time()
    if isinstance(model, DecisionTree):
        model.fit(data)
    elif isinstance(model, LogisticRegression_newton):
        X, y = data
        model.fit(X, y)
    else:
        raise Exception
    end = time.time()
    return end - start
