# 第五章 因子分析

## 01 因子分析简介
### 基本概念与正交因子模型
- **显变量 vs 隐变量**: 显变量可直接观测；隐变量（公共因子）由显变量相关结构导出，亦可用于检验区分效度
- **正交因子模型 (定义 5.2)**: $\boldsymbol{X} = \boldsymbol{A}\boldsymbol{F} + \boldsymbol{\mu} + \boldsymbol{\varepsilon}$，其中 $E(\boldsymbol{F})=\boldsymbol{0}, \operatorname{Cov}(\boldsymbol{F})=\boldsymbol{I}_m$，$E(\boldsymbol{\varepsilon})=\boldsymbol{0}, \operatorname{Cov}(\boldsymbol{\varepsilon})=\boldsymbol{\Psi}=\operatorname{diag}(\psi_1,\dots,\psi_p)$，$\operatorname{Cov}(\boldsymbol{F},\boldsymbol{\varepsilon})=\boldsymbol{0}$
- **协方差结构**: $\boldsymbol{\Sigma} = \boldsymbol{A}\boldsymbol{A}' + \boldsymbol{\Psi}$（标准化后 $\boldsymbol{R} = \boldsymbol{A}\boldsymbol{A}' + \boldsymbol{\Psi}$），非对角元完全由载荷矩阵 $\boldsymbol{A}$ 决定

### 统计意义与核心性质
- **因子载荷 ($a_{ij}$)**: $\operatorname{Cov}(\boldsymbol{X}, \boldsymbol{F}) = \boldsymbol{A}$；标准化后 $a_{ij} = \rho(X_i, F_j)$ 为变量与公共因子的相关系数
- **变量共同度 ($h_i$)**: $\sigma_{ii} = h_i + \psi_i$，其中 $h_i = \sum_{j=1}^m a_{ij}^2$ 反映公共因子对 $X_i$ 的方差贡献
- **公共因子方差贡献 ($q_j$)**: $q_j = \sum_{i=1}^p a_{ij}^2$，总方差贡献率为 $q_j / \sum_{i=1}^p \operatorname{Var}(X_i)$（标准化后为 $q_j/p$）
- **尺度不变性 (性质 5.1)**: $\boldsymbol{X}^* = \boldsymbol{C}\boldsymbol{X}$ 仍满足正交因子模型，公共因子 $\boldsymbol{F}$ 不变
- **载荷矩阵不唯一 (性质 5.2)**: 对任意 $m\times m$ 正交阵 $\boldsymbol{\Gamma}$，$\boldsymbol{A}^* = \boldsymbol{A}\boldsymbol{\Gamma}, \boldsymbol{F}^* = \boldsymbol{\Gamma}'\boldsymbol{F}$ 仍是等价模型（因子旋转基础）

## 02 参数估计
### 适用性检验与估计方法概览
- **适用性检验**: 相关系数 $|\rho| > 0.3$、样本量 $n \ge 5p$、KMO 检验 ($KMO > 0.5$)、Bartlett 球形检验（R: `kmo()`, `cortest.bartlett()`）
- **主成分法 (PCM)**: 截断特征值分解 $\widehat{\boldsymbol{A}} = (\sqrt{\lambda_1}\boldsymbol{p}_1, \dots, \sqrt{\lambda_m}\boldsymbol{p}_m)$，$\widehat{\psi}_i = \sigma_{ii} - \widehat{h}_i$，此时 $q_j = \lambda_j$
- **主因子法 (PFM)**: 对约相关矩阵 $\boldsymbol{R}^* = \boldsymbol{R} - \boldsymbol{\Psi}$ 作谱分解；初始估计 $h_i$ 可取 $1$、$1/\rho^{ii}$ 或 $\max_{j\ne i}|\rho_{ij}|$，并可迭代收敛
- **极大似然法 (MLE)**: 假定 $\boldsymbol{F}\sim N_m(\boldsymbol{0},\boldsymbol{I}), \boldsymbol{\varepsilon}\sim N_p(\boldsymbol{0},\boldsymbol{\Psi})$，在 $\boldsymbol{A}'\boldsymbol{\Psi}^{-1}\boldsymbol{A}$ 对角约束下迭代求解；R 中用 `factanal()`

## 03 因子旋转
### 正交旋转的不变性
- **旋转变换**: $\boldsymbol{A}^* = \boldsymbol{A}\boldsymbol{\Gamma}$（$\boldsymbol{\Gamma}$ 正交），使载荷趋向 $\pm 1$ 或 $0$ 以增强可解释性
- **不变量**: 正交旋转不改变变量共同度 $h_i$、$m$ 个因子的累计贡献率 $\operatorname{tr}(\boldsymbol{A}\boldsymbol{A}')$ 以及残差矩阵 $\boldsymbol{R} - (\boldsymbol{A}\boldsymbol{A}' + \boldsymbol{\Psi})$；会改变单个因子的贡献 $q_j$

### 常用旋转方法
- **最大方差法 (Varimax)**: 最大化各列归一化载荷平方 $d_{ij}^2 = a_{ij}^2/h_i$ 的方差之和 $V = \sum_{j=1}^m V_j$，通过 $C_m^2$ 次两两轴配对旋转迭代求解 (Kaiser, 1958)；R 中用 `varimax()`
- **其他方法**: 最大四次方法 (Quartimax)、最大平衡值法 (Equamax)

## 04 因子得分
### 两种估计方法
- **Bartlett 因子得分 (加权最小二乘)**: $\widehat{\boldsymbol{F}} = (\widehat{\boldsymbol{A}}'\widehat{\boldsymbol{\Psi}}^{-1}\widehat{\boldsymbol{A}})^{-1}\widehat{\boldsymbol{A}}'\widehat{\boldsymbol{\Psi}}^{-1}(\boldsymbol{X}-\bar{\boldsymbol{X}})$；主成分法不加权时 $\widehat{\boldsymbol{F}}^{PC} = (\widehat{\boldsymbol{A}}'\widehat{\boldsymbol{A}})^{-1}\widehat{\boldsymbol{A}}'(\boldsymbol{X}-\bar{\boldsymbol{X}})$ 为前 $m$ 主成分标准化
- **Thompson 因子得分 (回归法)**: 基于后验均值 $E(\boldsymbol{F}|\boldsymbol{X})$，得 $\widehat{\boldsymbol{F}^*} = \widehat{\boldsymbol{A}}'(\widehat{\boldsymbol{A}}\widehat{\boldsymbol{A}}'+\widehat{\boldsymbol{\Psi}})^{-1}(\boldsymbol{X}-\bar{\boldsymbol{X}}) = (\boldsymbol{I}+\widehat{\boldsymbol{A}}'\widehat{\boldsymbol{\Psi}}^{-1}\widehat{\boldsymbol{A}})^{-1}\widehat{\boldsymbol{A}}'\widehat{\boldsymbol{\Psi}}^{-1}(\boldsymbol{X}-\bar{\boldsymbol{X}})$

### 性质比较与分析步骤
- **无偏性 vs 均方误差**: Bartlett 得分无偏，Thompson 得分有偏；但 Thompson 预测均方误差 $(\boldsymbol{I}+\boldsymbol{A}'\boldsymbol{\Psi}^{-1}\boldsymbol{A})^{-1}$ 小于 Bartlett 的 $(\boldsymbol{A}'\boldsymbol{\Psi}^{-1}\boldsymbol{A})^{-1}$，有效性更优
- **因子分析五步流程**: 适用性检验 (KMO/Bartlett) $\to$ 参数估计 $\to$ 确定因子数 $m \to$ 因子旋转 $\to$ 模型解释与得分应用
