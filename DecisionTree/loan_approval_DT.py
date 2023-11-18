from Utils import *

# 导入数据集
# 划分数据集
train, test = loan_approval_data_processed(directory='../data/loan_approval_dataset.csv', minmax=False, scale=0.75)

# 训练模型
dt_model = DecisionTree()
dt_model.fit(train)

# 决策树可视化
dt_model.tree.visualize()

# 模型在测试集上的精度
accuracy = dt_model.accuracy(test)
print("模型的准确率为:%2.2f" % (accuracy * 100), "%")  # 模型的准确率为: 97.38 %

# 绘制学习曲线 learning curve
# 横轴：训练集的样本数量，纵轴：模型在训练集和测试集上的准确度
processed_data = loan_approval_data_processed(directory='../data/loan_approval_dataset.csv', minmax=False, split=False)
draw_learning_curve(processed_data, DecisionTree(), start=0.01, end=1.0, step=0.05)
