---
title: 决策树
date: 2023-11-07 18:37:02
tags: 机器学习
mathjax: true
---

# 决策树 decision tree

## 1. 简要介绍

决策树，英文为 decision tree。决策树是一种**分类**学习方法，基于**树结构**进行决策。

> ### 决策树的基本原理
>
> 1. 一般的，一棵决策树包含一个**根结点**、若干个**内部结点**和若干个**叶结点**；其中叶结点对应决策结果，其他每个结点对应一个属性测试；根结点包含样本全集，其他每个节点包含的样本集合根据属性测试的结果被划分到子结点中。
>
> 2. 决策树学习基本算法如下，
>
>    <img src="https://raw.githubusercontent.com/Nasir1423/blog-img/main/image-20231101123308690.png" alt="image-20231101123308690" style="zoom: 50%;" />
>
>    决策树的生成是一个**递归过程**，三种情形会导致递归返回，
>
>    - **当前结点包含的样本全属于同一类别**，无需划分
>    - **当前结点属性集为空 or 所有样本在所有属性上取值相同**，无法划分 ==> 将当前结点标记为叶结点，并将其类别设定为该结点所含样本最多的类别 (利用当前节点的后验分布)
>    - **当前结点包含的样本集合为空**，不能划分 ==> 将当前结点标记为叶结点，并且将其类别设定为其父结点所包含样本最多的类别 (将父结点的样本分布作为当前结点的先验分布)

> ### 属性划分方法
>
> 决策树学习的关键是如何选择**最优的划分属性**. 一般而言，随着划分过程不断进行，我们希望决策树的**分支结点所包含的样本尽可能属于同一类别**，即结点的"纯度" (purity) 越来越高. 关于选择最优的划分属性，主要有三种方法：信息增益、增益率基尼系数。
>
> ---
>
> #### Ⅰ信息增益（ID3 决策树，Iterative Dichotomiser）
>
> 1. “**信息熵**” (information entropy) 是**度量样本集合纯度**最常用的一种指标，当前**样本集合 D 的信息熵**定义如下 ( 假定当前样本集合 D 中第 $k$ 类样本所占的比例为 $p_k(k=1,2,...,|\gamma|)$ )
>    $$
>    Ent(D)=-\sum_{k=1}^{|\gamma|}p_klog_2p_k \tag{1}
>    $$
>
>
>    对于信息熵有以下结论
>
>    - **Ent(D) 的值越小，则 D 的纯度越高**
>    - 约定：若 $p=0$, 则 $plog_2p =0$
>    - $min\ Ent(D)=0$，$max\ Ent(D)=log_2|\gamma|$
>
> 2. “**信息增益**” (information gain) 可以用于**选择最优划分属性**，**属性 a 对样本集合 D 进行划分所获得的信息增益**定义如下( 假定离散属性 a 有 V 个可能的取值 {$a^1,a^2,...,a^V$}，若使用属性 a 对样本集合 D进行划分，则会产生 V 个分支结点，其中第 v 个分支结点包含了 D 中所有在属性 a 上取值为 $a^v$ 的样本，记为 $D^v$ )，
>    $$
>    Gain(D,a)=Ent(D)-\sum_{v=1}^V\frac{|D^v|}{D}Ent(D^v) \tag{2}
>    $$
>    其中，由于考虑到不同分支结点所包含的样本数不同，因此给分支结点赋予权重 $|D^v|/|D|$ ，即样本数越多的分支结点的影响越大。对于信息增益有以下结论
>
>    - **Gain(D,a) 越大，则表示使用属性 a 来进行划分所获得的 “纯度提升” 越大**
>
> 3. 信息增益准则的特点：信息增益准则**对可取值数目较多的属性有所偏好**
>
> 4. 最优划分属性的选择：从候选划分属性中**选择信息增益最大的属性**
>    $$
>    a_*=\underset{a\in A}{arg\ max}\ Gain(D,a)
>    $$
>
> ---
>
> #### Ⅱ 增益率（C4.5 决策树）
>
> 1. “**增益率**” (gain ration) 的定义如下，
>    $$
>    \begin{align}
>    Gain\_ratio(D,a)=\frac{Gain(D,a)}{IV(a)}\tag{3}\\
>    IV(a)=-\sum_{v=1}^V\frac{|D^v|}{|D|}log_2\frac{|D^v|}{|D|}\tag{4}\\
>    \end{align}
>    $$
>    其中，IV(a) 称之为属性 a 的固有值 (intriinsic value)，一般而言，属性 a 的可能取值数目越多 (V 越大)，则固有之 IV(a) 的值通常越大
>
> 2. 增益率准则的特点：增益率准则**对可取值数目较少的属性有所偏好**
>
> 3. 最优划分属性的选择：启发式
>
>    - 先从候选划分属性中**找到信息增益高于平均水平的属性**
>    - 再从中选择**增益率最高的**
>
> ---
>
> #### Ⅲ 基尼系数（CART 决策树，Classification and Regression BinaryTree）
>
> 1. “基尼值” (Gini) 可以用来**度量数据集的纯度**，**数据集 D 的基尼值**定义如下，
>    $$
>    \begin{align}
>    Gini(D)&=\sum_{k=1}^{|\gamma|}\sum_{k'\ne k}p_kp_{k'}\\
>    &=1-\sum_{k=1}^{|\gamma|}p_k^2\tag{5}
>    \end{align}
>    $$
>    其中，Gini(D) 表示从数据集 D 中随机抽取两个样本，其类别标记不一致的概率。对于基尼值，有以下结论，
>
>    - **Gini(D) 越小，则数据集 D 的纯度越高**
>    
> 2. “基尼指数” (Gini index) 用于**选择最优划分属性**，属性 a 的基尼指数定义为，
>    $$
>    Gini\_index(D,a)=\sum_{v=1}^V\frac{|D^v|}{D}Gini(D^v)\tag{6}
>    $$
>
> 3. 最优划分属性的选择：从候选划分属性中**选择基尼指数最小的属性**
>    $$
>    a_*=\underset{a\in A}{arg\ min}\ Gini\_index(D,a)
>    $$
>
> 

> ### 连续值处理
>
> 由于连续属性的可取值数目不再有限，因此不能直接根据其可取值对结点进行划分，于是可以采取**连续属性离散化**技术对连续属性进行处理。C4.5 决策树算法中采用的**二分法** (bi-partition) 对连续属性进行处理，其步骤如下，
>
> ---
>
> 给定样本集合 D 和连续属性 a，假定 a 在 D 上出现了 n 个不同的取值，将这些值从小到大进行排序，记为 {$a^1,a^2,...,a^n$}.
>
> 1. 求得**候选划分点集合 T**：基于 T 中的划分点 t 可以将 D 分为子集 $D_t^-$ 和 $D_t^+$，其中 $D_t^-=\{ 在属性 a 上取值 \le t 的样本 \}$，$$D_t^+=\{ 在属性 a 上取值 \gt t 的样本 \}$$。对于连续属性 a，可以求得包含 n-1 个元素的候选划分点集合，
>    $$
>    T_a=\{\frac{a^i+a^{i+1}}{2}|1\le i\le n-1\}
>    $$
>    即将区间 $[a^i,a^{i+1})$ 的中位点 $\frac{a^i+a^{i+1}}{2}$ 作为候选划分点.
>
> 
>
> 2. 基于信息增益**选取最优的划分点**：Gain(D,a,t) 是样本集合基于划分点 t 二分后的信息增益，最优划分点即使 Gain(D,a,t) 最大化的划分点
>    $$
>    \begin{align}
>    Gain(D,a)&=\underset{t\in T_a}{max}\ Gain(D,a,t)\\
>    &=\underset{t\in T_a}{max}\ Ent(D)-\underset{\lambda\in\{-,+\}}{\sum}\frac{|D_t^\lambda|}{|D|}Ent(D_t^\lambda)
>    \end{align}
>    $$
>
> ---
>
> 注意：**若当前结点划分属性为连续属性，则该属性还可以作为其后代的划分属性**

## 2. 模型训练步骤

><img src="https://raw.githubusercontent.com/Nasir1423/blog-img/main/image-20231101123308690.png" alt="image-20231101123308690" style="zoom: 50%;" />

## 3. 数据集介绍

DATASET: [Loan-Approval-Prediction-Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset/data), **4296 × 13 features** (9 integer, 2 string, 1 id, 1 other)

The loan approval dataset is a collection of financial records and associated information used to **determine the eligibility of individuals or organizations for obtaining loans** from a lending institution. It includes various factors such as **cibil score, income, employment status, loan term, loan amount, assets value, and loan status**. 
