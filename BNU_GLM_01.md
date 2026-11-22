# 第1章 绪论 (广义线性模型)

## 1.1 统计学及其思维模式
- **统计学定义**: 通过收集与分析数据来认识未知现象的科学
- **研究闭环**: 提出问题 $\to$ 收集分析数据 $\to$ 得到认识 $\to$ 实践检验发现不足 $\to$ 下一步研究（拒绝“迷信”）

## 1.2 未知现象的数学描述
### 理想模型与现实模型
- **理想模型三种等价表达**: $\mathbf{Y} = \mathbf{f}(\mathbf{X}) + \boldsymbol{\epsilon}$ ($E(\boldsymbol{\epsilon}|\mathbf{X})=0$) $\iff \mathbb{E}(\mathbf{Y}|\mathbf{X}) = \mathbf{f}(\mathbf{X}) \iff$ 条件分布族 $\mathcal{F} = \{F_{\mathbf{X}}\}$
- **定理 1.2.1**: 均方误差最小准则下，理想预报公式 $\mathbf{Y} = \mathbf{f}(\mathbf{X})$ 最优
- **现实模型**: 未知 $\mathbf{f}$ 时用参数函数族 $F = \{\mathbf{f}(\cdot|\theta): \theta \in \Theta\}$ 逼近，得 $\mathbf{Y} = \mathbf{f}(X_1, \dots, X_k|\theta_0) + \eta$
### 现实模型的参数估计与模型选择
- **最小二乘估计**: $\hat{\theta} = \arg\min_\theta Q(\theta)$，残差均方 $\hat{\sigma}^2 = Q(\hat{\theta})/(n-p)$
- **父模型 vs 子模型**: 父模型残差平方和更小，但含无关变量时新样本预报变差，可用 `step()` 逐步回归筛选

## 1.3 线性回归模型的定义与编码
### 定义与多项式逼近
- **一般定义**: $\mathbb{E}(Y|\mathbf{X}) = \mathbf{Z}(\mathbf{X})\boldsymbol{\beta}^\top$，$\mathbf{Z}(\mathbf{X})$ 为设计向量，$\mathbf{Z}(\mathbf{X})\boldsymbol{\beta}^\top$ 为线性预测子
- **逼近应用**: 利用幂函数族 $1, x, x^2, \dots$ 逼近连续非线性回归函数
### 离散变量编码（等价模型族，参数含义不同）
- **哑变量编码 (Dummy)**: $q$ 类转为 $q-1$ 维 $(1_{\{x_2\}}(X), \dots, 1_{\{x_q\}}(X))$，截距为参考类均值，系数为与参考类之差
- **效应编码 (Effect)**: $(1_{\{1\}}(X)-1_{\{k\}}(X), \dots)$，对比各类相对于总体/参考类的效应
- **独热编码 (One-hot)**: $(1_{\{x_1\}}(X), \dots, 1_{\{x_q\}}(X))$，无截距项时系数直接为各类均值

## 1.4 广义线性模型 (GLM)
### 1.4.1 一变量广义线性模型
- **模型形式**: $\mathbb{E}(Y|\mathbf{X}) = h(\mathbf{Z}(\mathbf{X})\boldsymbol{\beta}^\top)$ 或 $g(\mathbb{E}(Y|\mathbf{X})) = \mathbf{Z}(\mathbf{X})\boldsymbol{\beta}^\top$
- **核心组件**: 响应函数 $h$、链接函数 $g = h^{-1}$、设计向量 $\mathbf{Z}(\mathbf{X})$、线性预测子 $\mathbf{Z}(\mathbf{X})\boldsymbol{\beta}^\top$
### 1.4.2 R 函数 `glm()` 与常用链接函数
- **函数调用**: `glm(formula, family = familyfun(link = "..."), data)`
- **常用分布族**: `gaussian()` (默认 `identity`)、`binomial()` (默认 `logit`，可选 `probit`, `cloglog`, `cauchit`)、`poisson()` (默认 `log`，可选 `sqrt`)
### 1.4.3 多变量 GLM 与 1.4.4 `vglm()`
- **$q$ 变量 GLM**: $\mathbb{E}(\mathbf{Y}|\mathbf{X}) = \mathbf{h}(\mathbf{Z}(\mathbf{X})\boldsymbol{\beta}^\top)$，$\mathbf{Z}$ 为 $q \times p$ 设计矩阵
- **VGAM 包 `vglm()`**: `vglm(cbind(y1, y2) ~ x1 + x2, family = poissonff(), data)`，构造分块对角设计矩阵
### 1.4.5 多分类广义线性模型
- **建模困境**: 无序分类量化为整数后期望无意义
- **哑变量转化**: $\mathbf{Y} = (1_{\{y_1\}}(Y), \dots, 1_{\{y_q\}}(Y))$ ($q=k-1$)，转为条件概率向量 $\mathbb{E}(\mathbf{Y}|\mathbf{X}) = \mathbf{h}(\mathbf{Z}(\mathbf{X})\boldsymbol{\beta}^\top)$

## 1.5 R Markdown 与课程安排
- **R Markdown**: 实现“代码-计算-排版”一体化，用于完成课程作业
- **后续章节**: 第 2 章一变量 GLM，第 3 章 $k$ 分类 GLM ($k>2$)
