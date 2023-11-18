# 对数几率回归 logit regression

## 1. 简要介绍

逻辑回归，又称对数几率回归，英文为 logistic regression，或 logit regression。对数几率回归是一种**分类**学习方法，该方法直接对分类的可能性进行建模，无需事先假设数据分布，因此可以避免假设分布不准确带来的问题。

> 1. 对于**二分类**任务，即输出标记为 $y\in${$0,1$}，因为**线性回归模型** $z = w^Tx+b\in R$， 所以需要引进一个**替代函数** $g^-(·)$ ，得到一个**广义的线性回归模型** $y=g^-(z)=g^-(w^Tx+b)\in(0,1)$，即 $g(y)$ 与 $z = w^Tx+b$ 之间是线性关系。
>
> 2. 该模型满足，当 $z\rightarrow \infty$ 时，$y\rightarrow 1$，当 $z\rightarrow -\infty$ 时，$y\rightarrow 0$，即引入了一种**概率关系**，当 p 越大，则正例的可能性更大，反例的可能性更小。
>
> 3. 通常来说，**对数几率函数** (logistic function) 作为**任意阶可导的凸函数**，也是一个常用的替代函数，
>
> $$
> y=\frac{1}{1+e^{-z}} \tag{1}
> $$
>
> <img src="https://raw.githubusercontent.com/Nasir1423/blog-img/main/image-20231027210529577.png" alt="image-20231027210529577" style="zoom: 50%;" />
>
> 

> 1. 基于对数几率函数可以导出**广义线性回归模型**，该模型又称为**对数几率回归模型**。
>
> $$
> ln\frac{y}{1-y}=w^Tx+b \tag{2}
> $$
>
> 2. 若将 y 视为样本 x 是正例的可能性，1-y 视为样本 x 是反例的可能性，则二者比值反映了 x 是正例的相对可能性，称之为**几率** (odds)，因此上述导出的广义线性回归模型又可以称为**对数几率** (log odds 或 logit)。
>
> $$
> odds = \frac{y}{1-y} \tag{3}
> $$
>
> 3. 基于以上观点，为了求解对数几率回归模型，即**需要采用一定的方法估计参数 $w, b$**，无妨记 $\beta=(w;b)$, $\hat{x}=(x;1)$，则有 $z = w^Tx+b=\beta^Tx$。
>
> 4. 如将 $y$ 视为类后验概率估计 $p(y=1|x)$, $1-y$ 视为 $p(y=0|x)$，则式 (2) 可以表示为，
>    $$
>    ln\frac{p(y=1|x)}{p(y=0|x)}=w^Tx+b \tag{4}
>    $$
>    进一步计算得到正例和反例的**类后验概率估计**为，
>    $$
>    \begin{align}
>    p(y=1|x)=\frac{e^{w^Tx+b}}{1+e^{w^Tx+b}}=p_1(\hat{x};\beta)=p(y=1|\hat{x};\beta) \tag{5-1} \\
>    p(y=0|x)=\frac{1}{1+e^{w^Tx+b}}=p_0(\hat{x};\beta)=p(y=0|\hat{x};\beta) \tag{5-2}\\
>    \end{align}
>    $$
>
> $$
> p_1(\hat{x};\beta)+p_0(\hat{x};\beta)=1
> $$

> 1. 对数几率回归模型的求解在于参数 $w,b$ 或 $\beta=(w;b)$ 的求解，因此基于已有样本数据，可以采用**极大似然法** (maximum likelihood method) 来进行参数估计，即**令每个样本属于其真实标记的概率越大越好**。
>
> 2. 给定数据集 $\left \{ \left ( x_i,y_i \right ) \right \}^m_{i=1}$，可以得到对数几率回归模型的**对数似然** (log likelihood)，优化方向是使其最大化，
>    $$
>    max:log\ likelihood=l(w,b)=\sum_{i=1}^{m}ln\ p(y_i|x_i;w,b)=\sum_{i=1}^{m}ln\ p(y_i;\hat{x_i},\beta) \tag{6}
>    $$
>
>
> 3. 对数似然中的**似然项** (likelihood) 可以改写如下，
>    $$
>    p(y_i|x_i;w,b)=p(y_i;\hat{x_i},\beta)=y_ip_1(\hat{x_i},\beta)+(1-y_i)p_0(\hat{x_i},\beta) \tag{7}
>    $$
>    故而**对数似然** 基于式 (5), (6), (7)可以改写如下，
>    $$
>    \begin{align}max:l(\beta)&=&\sum_{i=1}^nln[y_ip_1(\hat{x_i},\beta)+(1-y_i)p_0(\hat{x_i},\beta)] \tag{8} \\
>    &=&\sum_{i=1}^n[y_i\beta^T\hat{x}-ln(1+e^{\beta^T\hat{x}})]\end{align}
>    $$
>    式 (8) 的**最大化**式转换为**最小化**式如下，该式是关于 $\beta$ 的高阶可导的连续凸函数，
>    $$
>    min:l(\beta)=\sum_{i=1}^n[-y_i\beta^T\hat{x}+ln(1+e^{\beta^T\hat{x}})]\tag{9}
>    $$
>    根据式 (9) 及**数值优化算法** (梯度下降法 GDM，牛顿法 NM等) 可以求对数几率回归模型的最优解 $\beta^*$，
>    $$
>    \beta^*=\underset{\beta}{arg\ min\ l(\beta)}\tag{10}
>    $$

> 1. **牛顿法**求解对数几率回归模型步骤如下
>
>    <img src="https://raw.githubusercontent.com/Nasir1423/blog-img/main/image-20231028101813918.png" alt="image-20231028101813918" style="zoom: 50%;" />
>
>    其中**迭代解的更新公式**为
>    $$
>    \beta^{t+1}=\beta^t-t(\frac{\partial ^2l(\beta)}{\partial \beta \partial \beta^T})^{-1}\frac{\partial l(\beta)}{\partial \beta}\tag{11}
>    $$
>    关于 $\beta$ 的**一阶、二阶导数**为
>    $$
>    \begin{align}
>    \frac{\partial l(\beta)}{\partial \beta}=-\sum_{i=1}^n\hat{x_i}(y_i-p_1)\tag{12-1}\\
>    \frac{\partial^2 l(\beta)}{\partial \beta\partial \beta^T}=\sum_{i=1}^n\hat{x_i}\hat{x_i}^Tp_1p_0\tag{12-2}
>    \end{align}
>    $$
>
> 2. 牛顿法中的学习率 t 可以使用 **backtracking line search** 方法求解
>
>    <img src="https://raw.githubusercontent.com/Nasir1423/blog-img/main/image-20231028103654235.png" alt="image-20231028103654235" style="zoom:50%;" />

> 当海森矩阵非正定时，牛顿法失效，因此也可以采用**梯度下降法**求解参数  $\beta$，其步骤如下，
>
> <img src="https://raw.githubusercontent.com/Nasir1423/blog-img/main/20231107183519.png" style="zoom: 33%;" />

## 2. 模型训练步骤

dataset: $X=\left \{x_1,x_2,x_3,...,,x_n  \right \}, x_i=\left [ x_{i1},x_{i2},x_{i3},...,x_{im} \right ]^T$ (n samples × m features)

denote dataset as: $X=\left \{\hat{x_1},\hat{x_2},\hat{x_3},...,,\hat{x_n}  \right \}, \hat{x_i}=\left [ x_{i1},x_{i2},x_{i3},...,x_{im},1 \right ]^T$; $\beta=[w;b]$

> ### 牛顿法
>
> given a starting point $\beta$, tolorance $\varepsilon=10^{-6}$
>
> repeat 1、2、3、4
>
> 1. 计算牛顿步长 $\Delta \beta_{nt}$ 和牛顿减量 $\lambda^2$
>    $$
>    \begin{align}
>    newton\ stepsize:\Delta \beta_{nt}:=-(\frac{\partial ^2l(\beta)}{\partial \beta \partial \beta^T})^{-1}\frac{\partial l(\beta)}{\partial \beta}\\
>    newton\ decrement:\lambda^2:=(\frac{\partial l(\beta)}{\partial \beta})^{T}(\frac{\partial ^2l(\beta)}{\partial \beta \partial \beta^T})^{-1}\frac{\partial l(\beta)}{\partial \beta}\\
>    \frac{\partial l(\beta)}{\partial \beta}=-\sum_{i=1}^n\hat{x_i}(y_i-p_1)\\
>    \frac{\partial^2 l(\beta)}{\partial \beta\partial \beta^T}=\sum_{i=1}^n\hat{x_i}\hat{x_i}^Tp_1p_0
>    \end{align}
>    $$
>    
> 2. 停止准则 (当牛顿减量特别小的时候，表示函数此时十分平滑了)
>    $$
>    if\ \lambda^2/2\leq \varepsilon, then\ quit
>    $$
>
> 3. 回溯直线搜索，计算学习率 $t$
>
>    given $\hat{\alpha}\in(0,0.5), \hat{\beta}\in(0,1)$，其中 $\beta$ 和 $\hat{\beta}$ 代表不同含义
>    $$
>    \begin{align}
>    while\ f(\beta^k+t^k{\Delta \beta_{nt}}^k)>f(\beta^k)-\hat{\alpha} t^k\lambda^2\\
>    \ \ t^k:=\hat{\beta}t^k
>    \end{align}
>    $$
>
>
> 4. 参数更新
>    $$
>    \beta^{k+1}:=\beta^{k}+t^k{\Delta \beta_{nt}}^k
>    $$
>
> ### 梯度下降法
>
> given a starting point $\beta$，$\eta=10^-5$
>
> repeat 1、2、3、4
>
> 1. 计算梯度下降步长
>    $$
>    \Delta x:=-\frac{\partial l(\beta)}{\partial \beta}=\sum_{i=1}^n\hat{x_i}(y_i-p_1)
>    $$
>
> 2. 停止准则判断：exit if 
>    $$
>    ||\frac{\partial l(\beta)}{\partial \beta}||_2\le\eta
>    $$
>
>
> 3. 回溯直线搜索，计算学习率 $t$
>
>    given $\hat{\alpha}\in(0,0.5), \hat{\beta}\in(0,1)$，其中 $\beta$ 和 $\hat{\beta}$ 代表不同含义
>    $$
>    \begin{align}
>    while\ f(\beta^k+t^k{\Delta \beta_{nt}}^k)>f(\beta^k)-\hat{\alpha} t^k\lambda^2\\
>    \ \ t^k:=\hat{\beta}t^k
>    \end{align}
>    $$
>
> 4. 参数更新
>    $$
>    \beta^{k+1}:=\beta^{k}+t^k{\Delta \beta_{nt}}^k
>    $$

## 3. 数据集介绍

DATASET: [Loan-Approval-Prediction-Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset/data), **4296 × 13 features** (9 integer, 2 string, 1 id, 1 other)

The loan approval dataset is a collection of financial records and associated information used to **determine the eligibility of individuals or organizations for obtaining loans** from a lending institution. It includes various factors such as **cibil score, income, employment status, loan term, loan amount, assets value, and loan status**. 

