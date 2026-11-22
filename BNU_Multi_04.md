# 第四章 主成分分析

## 01 简介 (Introduction)
### 背景与动机
- **维数灾难**: 高维数据 ($p \gg n$) 导致计算复杂度剧增、过拟合及最近邻距离趋同 (Beyer et al. 99)
- **马氏距离与旋转**: 欧氏距离忽略变量相关性；正交旋转消除相关性并归一化方差得 $D_M = (\boldsymbol{x}-\boldsymbol{\mu})'\boldsymbol{\Sigma}^{-1}(\boldsymbol{x}-\boldsymbol{\mu})$
- **PCA 核心思想**: 线性降维工具，寻找方差最大且互不相关的正交投影方向，用少量综合变量反映绝大部分信息

### 主成分定义 (定义 4.1)
- **线性变换**: $Z_i = \boldsymbol{a}_i'\boldsymbol{X} = a_{1i}X_1 + \dots + a_{pi}X_p$
- **约束与目标**: 单位长度 $\boldsymbol{a}_i'\boldsymbol{a}_i = 1$；互不相关 $\boldsymbol{a}_i'\boldsymbol{\Sigma}\boldsymbol{a}_j = 0 \ (j<i)$；最大化方差 $\operatorname{Var}(Z_i) = \max \boldsymbol{a}'\boldsymbol{\Sigma}\boldsymbol{a}$

## 02 总体的主成分，性质及计算
### 数学推导与求解 (引理 4.1 & 定理 4.1–4.2)
- **瑞利商极值 (引理 4.1)**: 对称阵 $\boldsymbol{A}$ 的 $\max_{\|\boldsymbol{x}\|=1} \boldsymbol{x}'\boldsymbol{A}\boldsymbol{x} = \lambda_1$（在 $\boldsymbol{r}_1$ 处取到），依次正交得 $\lambda_{k+1}$
- **主成分求法**: 设 $\boldsymbol{\Sigma}$ 特征值 $\lambda_1 \ge \dots \ge \lambda_p \ge 0$，单位正交特征向量 $\boldsymbol{a}_1, \dots, \boldsymbol{a}_p$，则第 $i$ 主成分 $Z_i = \boldsymbol{a}_i'\boldsymbol{X}$
- **几何与代数意义**: 代数上将 $\boldsymbol{\Sigma}$ 对角化；几何上将原坐标系正交旋转至样本散布最分开的方向

### 总体主成分的性质
- **方差与不相关性**: $\operatorname{Var}(Z_i) = \lambda_i$，$\operatorname{Cov}(Z_i, Z_j) = 0 \ (i\ne j)$，$\operatorname{Cov}(\boldsymbol{Z}) = \boldsymbol{\Lambda} = \operatorname{diag}(\lambda_1,\dots,\lambda_p)$
- **总方差守恒**: $\sum_{i=1}^p \lambda_i = \sum_{i=1}^p \sigma_{ii} = \operatorname{tr}(\boldsymbol{\Sigma})$
- **因子载荷**: $\rho(Z_k, X_i) = \sqrt{\lambda_k}a_{ik}/\sqrt{\sigma_{ii}}$，满足 $\sum_{k=1}^p \rho^2(Z_k, X_i) = 1$ 及 $\sum_{i=1}^p \sigma_{ii}\rho^2(Z_k, X_i) = \lambda_k$

### 选取指标与标准化
- **贡献率与累积贡献率**: 单项 $\omega_k = \lambda_k/\sum \lambda_i$；前 $m$ 项累积 $f_m = \sum_{k=1}^m \lambda_k / \sum_{i=1}^p \lambda_i$（常用阈值 $85\%$）
- **变量方差贡献率**: $\nu_i^{(m)} = \sum_{k=1}^m \lambda_k a_{ik}^2 / \sigma_{ii}$
- **基于标准化 ($X_i^*$)**: 量纲或方差差异巨大时，从相关矩阵 $\boldsymbol{R}$ 求主成分；此时 $\sum_{i=1}^p \lambda_i^* = p$，载荷 $\rho(Z_k^*, X_i^*) = a_{ik}^*\sqrt{\lambda_k^*}$

## 03 样本主成分的性质及计算
### 样本统计量与求解
- **样本协方差阵与相关阵**: $\boldsymbol{S} = \frac{1}{n-1}\sum_{i=1}^n (\boldsymbol{X}_{(i)}-\bar{\boldsymbol{X}})(\boldsymbol{X}_{(i)}-\bar{\boldsymbol{X}})'$，$\widetilde{\boldsymbol{R}} = \hat{\boldsymbol{D}}^{-1/2}\boldsymbol{S}\hat{\boldsymbol{D}}^{-1/2}$
- **得分与载荷**: 第 $i$ 样本主成分 $\hat{Z}_i = \hat{\boldsymbol{a}}_i'\boldsymbol{X}$；中心化得分 $\hat{Z}_{ik} = \hat{\boldsymbol{a}}_i'(\boldsymbol{X}_{(k)}-\bar{\boldsymbol{X}})$，均值为 $0$
- **样本方差性质**: $s_{Z_i}^2 = \hat{\lambda}_i$，$\operatorname{Cov}(\hat{Z}_i, \hat{Z}_j) = 0$，$\sum_{i=1}^p s_{ii} = \sum_{i=1}^p \hat{\lambda}_i$

## 04 PCA 的应用与局限性
### 主成分保留个数与局限性
- **保留个数准则**: 累积贡献率阈值（如 $85\%$）、Kaiser 准则（相关阵 $\lambda_m \ge 1$）、碎石图 (Scree Plot) 拐点、可解释性
- **PCA 局限性**: 仅限线性正交变换（非线性需 Kernel PCA）；仅依赖均值和协方差阵；无监督（不适合类别分离方向与方差最大方向垂直的分类任务，如反例需用 LDA）；受异常点影响大、无尺度不变性

### R 语言实现与典型应用
- **`princomp()` vs `prcomp()`**: `princomp(data, cor=TRUE)` 基于特征分解（需 $n>p$）；`prcomp(data, scale=TRUE)` 基于 SVD（允许 $p>n$）
- **辅助函数**: `loadings()`、`predict()`、`screeplot()`、`biplot()`
- **应用场景**: 探索性数据分析与降维可视化（如 Iris 数据集）、数据压缩与图像重建、综合评价函数 $F = \sum a_k Z_k$
