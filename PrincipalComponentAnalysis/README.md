# 主成分分析

## 1. 原理概述

1. 主成分分析（Principal Component Analysis，简称 PCA），其主要目标是**将高维数据映射到低维空间中**，并且希望在该低维空间中**原数据的信息量尽可能不丢失**。其中该低维空间中的特征是**全新的正交特征**（即线性无关），又称为**主成分**。

2. 为了在低维空间中尽量保留高维数据的信息，则对原始特征降维时，需要尽可能**在原始特征具有最大投影信息量的维度上进行投影**，从而使得降维后的信息量损失最小。

3. 为了实现上述要求，PCA 的工作内容就是**从原始的空间中顺序地找出一组相互正交的坐标轴（特征）**

   - 第一个坐标轴（特征）选取**原始数据中方差最大的方向**，对应于**第一主成分（PC1）**。
   - 第二个坐标轴选取**与第一个坐标轴正交的平面中原始数据方差最大的方向**，对应于**第二主成分（PC2）**。
   - 第三个坐标轴选取**与第一、第二个坐标轴正交的平面中原始数据方差最大的方向**，对应**第三主成分（PC3）**。
   - 以此类推

4. 通过数学分析，可以发现：**原始数据在 PC1 上的投影上方差最大，代表了绝大部分信息**，新的主成分所包含的信息量依次递减。因此**大部分方差包含在前 k 个坐标轴（主成分）中，后边的坐标轴（主成分）所含的方差几乎为 0**，通过此我们可以只保留包含绝大部分方差的特征维度，而忽略包含方差几乎为 0 的特征维度，从而实现对数据特征的降维处理。

   > i.e. **（全新的正交）特征、主成分、坐标轴在这里为统一概念**，PCA 主要功能就是将高纬特征映射到低维的正交特征（又称之为主成分）

## 2. 最大方差理论（PCA 的实现步骤）

1. 在信号处理领域，我们认为**信号具有较大方差，而噪声具有较小方差**。主成分分析（Principal Component Analysis，PCA）是一种常用于数据降维的技术，其**核心思想是寻找数据中变异性最大的方向（主成分）**，以便通过选择少量主成分来近似表示原始数据。因此我们不难引出 PCA 的目标即**最大化投影方差**，也就是让**数据在主轴上投影的方差最大**。
2. **最大方差理论**的关键观点是，通过选择方差最大的方向，我们能够最大限度地保留原始数据的信息。这是因为**方差衡量了数据在某个方向上的变异性**，而选择方差最大的方向相当于选择了数据中最主要的变化方向。

> 主成分分析（PCA）的实现步骤如下，
>
> 1. 对所有**样本进行中心化**：对原始数据进行中心化，可以移除数据的均值，使得数据围绕原点分布。从而有利于 PCA 基于数据的相对位置进行降维。
>
> 2. **计算样本的协方差矩阵 $XX^T$**：样本 $x_i$ 在投影方向 $W$ （单位向量，unit vector）上的投影坐标为 $x_i^TW$，因此所有样本点投影后的方差可以表示为 $D(x)=\sum_iW^Tx_ix_i^TW$（等价于 $\underset{W}{max}\ W^TXX^TW$），且 $W^TW=I$。因此优化问题变为，
>    $$
>    \underset{W}{max}\ W^TXX^TW\tag{1}\\
>    s.t.\ W^TW=I
>    $$
>
> 3. **对协方差矩阵 $XX^T$ 做特征值分解**，解得特征值和特征向量：对式 (1) 使用拉格朗日乘子法有 $XX^TW=\lambda W$，将其带入 $D(x)$ 得到 $D(x)=W^TXX^TW=W^T\lambda W=\lambda$，即**样本点投影后的方差即协方差矩阵 $XX^T$ 的特征值**，且最大方差即协方差矩阵最大的特征值。
>
> 4. **选择主成分**：按照特征值**降序排列**特征向量，选择前几个特征向量作为主成分。这些主成分构成了一个新的坐标系，称之为主成分空间。关于主成分数量选择有以下两种方式：
>
>    - **预先指定**低维空间的维数为 d'，则取最大的 d' 个特征值对应的特征向量 $w_1,w_2,w_3,..,w_d'$ 即可
>
>    - **设置重构阈值**（可看作累计解释方差），记为 t，然后取使得下式成立的最小的 d' 值，再取对应数量的特征向量即可
>      $$
>      \frac{\sum_{i=1}^{d'}\lambda_i}{\sum_{i=1}^{d}\lambda_i}\ge t\tag{2}
>      $$
>      
>
> 5. **数据投影**：将原始数据投影到选择的主成分空间中，得到降维后的数据。

