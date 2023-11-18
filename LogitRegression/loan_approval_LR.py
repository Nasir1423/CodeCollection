from LoanApprovalPredict.model.LogisticRegression.LogisticRegression_newton import LogisticRegression_newton
from Utils import *

# 导入数据，并且获得预处理过的训练集和测试集合
train, test = loan_approval_data_processed(directory='../data/loan_approval_dataset.csv', minmax=True, scale=0.75)

# 划分训练集和测试集的样本数据、标签
X_train, y_train = dataset_split(train)
X_test, y_test = dataset_split(test)

# 模型训练
lr_model = LogisticRegression_newton()
lr_model.fit(X_train, y_train)

# 模型预测精度测试
accuracy = lr_model.accuracy(X_test, y_test)
print("模型的准确率为:%2.2f" % (accuracy * 100), "%")  # 模型的准确率为:91.92 %

# 绘制学习曲线 learning curve
# 横轴：训练集的样本数量，纵轴：模型在训练集和测试集上的准确度
processed_data = loan_approval_data_processed(directory='../data/loan_approval_dataset.csv', minmax=True, split=False)
draw_learning_curve(processed_data, LogisticRegression_newton())



