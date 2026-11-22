# Chapter 3 多元正态分布

## 01 多元分布
### 随机向量、分布函数与独立性
- **联合分布函数**: $F(\boldsymbol{x}) = P(X_1 \le x_1, \dots, X_p \le x_p)$，边缘分布 $F_{\boldsymbol{X}^{(q)}}(x_1,\dots,x_q) = F(x_1,\dots,x_q,\infty,\dots,\infty)$
- **连续型密度函数**: $F(\boldsymbol{x}) = \int_{-\infty}^{x_1}\cdots\int_{-\infty}^{x_p} f(t_1,\dots,t_p)\,dt_1\cdots dt_p$，边缘密度对多余分量积分
- **独立性**: $\boldsymbol{X}, \boldsymbol{Y}$ 独立 $\Leftrightarrow F(\boldsymbol{x},\boldsymbol{y}) = F_{\boldsymbol{X}}(\boldsymbol{x})F_{\boldsymbol{Y}}(\boldsymbol{y}) \Leftrightarrow f(\boldsymbol{x},\boldsymbol{y}) = f_{\boldsymbol{X}}(\boldsymbol{x})f_{\boldsymbol{Y}}(\boldsymbol{y})$；相互独立 $\implies$ 两两独立（反之不成立）

### 特征函数 ($\varphi_{\boldsymbol{X}}(\boldsymbol{t})$)
- **定义**: 向量 $\varphi_{\boldsymbol{X}}(\boldsymbol{t}) = E[e^{i\boldsymbol{t}'\boldsymbol{X}}]$；矩阵 $\varphi_{\boldsymbol{X}}(\boldsymbol{T}) = E[e^{\operatorname{tr}(i\boldsymbol{T}'\boldsymbol{X})}]$
- **核心性质**: $\varphi_{\boldsymbol{X}}(\boldsymbol{0}) = 1, |\varphi|\le 1$；与分布函数一一对应；$k$ 阶矩 $\varphi_X^{(k)}(0) = i^k E(X^k)$
- **独立与线性变换**: 独立 $\Leftrightarrow \varphi(\boldsymbol{t},\boldsymbol{s}) = \varphi_{\boldsymbol{X}}(\boldsymbol{t})\varphi_{\boldsymbol{Y}}(\boldsymbol{s})$；$\boldsymbol{Y}=\boldsymbol{A}\boldsymbol{X}+\boldsymbol{a} \Rightarrow \varphi_{\boldsymbol{Y}}(\boldsymbol{t}) = e^{i\boldsymbol{t}'\boldsymbol{a}}\varphi_{\boldsymbol{X}}(\boldsymbol{A}'\boldsymbol{t})$

### 均值、协方差阵与相关阵
- **均值线性性**: $E(\boldsymbol{A}\boldsymbol{X}\boldsymbol{B}+\boldsymbol{C}) = \boldsymbol{A}E(\boldsymbol{X})\boldsymbol{B}+\boldsymbol{C}$，$E(\operatorname{tr}(\boldsymbol{A}\boldsymbol{X})) = \operatorname{tr}(\boldsymbol{A}E(\boldsymbol{X}))$
- **协方差阵 ($\boldsymbol{\Sigma}$)**: $\operatorname{Cov}(\boldsymbol{X}) = E[(\boldsymbol{X}-\boldsymbol{\mu})(\boldsymbol{X}-\boldsymbol{\mu})'] \ge \boldsymbol{0}$（对称非负定）；$\operatorname{Cov}(\boldsymbol{A}\boldsymbol{X},\boldsymbol{B}\boldsymbol{Y}) = \boldsymbol{A}\operatorname{Cov}(\boldsymbol{X},\boldsymbol{Y})\boldsymbol{B}'$
- **相关矩阵 ($\boldsymbol{R}$)**: $\rho_{ij} = \sigma_{ij}/\sqrt{\sigma_{ii}\sigma_{jj}}$，与标准差阵 $\boldsymbol{V}^{1/2}$ 满足 $\boldsymbol{\Sigma} = \boldsymbol{V}^{1/2}\boldsymbol{R}\boldsymbol{V}^{1/2}$
- **二次型期望 (定理 3.1)**: $E(\boldsymbol{X}'\boldsymbol{A}\boldsymbol{X}) = \operatorname{tr}(\boldsymbol{A}\boldsymbol{\Sigma}) + \boldsymbol{\mu}'\boldsymbol{A}\boldsymbol{\mu}$
- **一元正态矩 ($N(\mu,\sigma^2)$)**: 奇数阶中心距 $B_{2i+1}=0$，偶数阶 $B_{2i}=(2i-1)!!\sigma^{2i}$

## 02 多元正态分布定义及性质
### 四种等价定义 ($\boldsymbol{X} \sim N_p(\boldsymbol{\mu}, \boldsymbol{\Sigma})$)
- **定义 3.11 (线性变换)**: $\boldsymbol{X} = \boldsymbol{A}_{p\times q}\boldsymbol{Y} + \boldsymbol{\mu}$，其中 $\boldsymbol{Y} \sim N_q(\boldsymbol{0}, \boldsymbol{I})$，$\boldsymbol{\Sigma} = \boldsymbol{A}\boldsymbol{A}'$
- **定义 3.12 (特征函数)**: $\varphi_{\boldsymbol{X}}(\boldsymbol{t}) = \exp(i\boldsymbol{t}'\boldsymbol{\mu} - \frac{1}{2}\boldsymbol{t}'\boldsymbol{\Sigma}\boldsymbol{t})$（$\boldsymbol{\Sigma} \ge \boldsymbol{0}$）
- **定义 3.13 (线性组合)**: 对任意 $\boldsymbol{a} \ne \boldsymbol{0}$，$\boldsymbol{a}'\boldsymbol{X}$ 服从一元正态分布（等价定理 3.2）
- **定义 3.14 (密度函数, $\boldsymbol{\Sigma}>\boldsymbol{0}$)**: $f(\boldsymbol{x}) = (2\pi)^{-p/2}|\boldsymbol{\Sigma}|^{-1/2}\exp\left(-\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})'\boldsymbol{\Sigma}^{-1}(\boldsymbol{x}-\boldsymbol{\mu})\right)$

### 核心性质
- **线性变换封闭性**: $\boldsymbol{Z} = \boldsymbol{B}\boldsymbol{X} + \boldsymbol{\theta} \sim N_q(\boldsymbol{B}\boldsymbol{\mu}+\boldsymbol{\theta}, \boldsymbol{B}\boldsymbol{\Sigma}\boldsymbol{B}')$
- **边缘分布**: 任意子集仍服从多元正态分布（反之不成立，边缘正态不能推出联合正态）
- **独立求和**: 独立正态向量之和 $\sum \boldsymbol{X}_i \sim N_p(\sum\boldsymbol{\mu}_i, \sum\boldsymbol{\Sigma}_i)$
- **二次型分布与标准化**: $(\boldsymbol{X}-\boldsymbol{\mu})'\boldsymbol{\Sigma}^{-1}(\boldsymbol{X}-\boldsymbol{\mu}) \sim \chi_p^2$；$\boldsymbol{\Sigma}^{-1/2}\boldsymbol{X} \sim N_p(\boldsymbol{\Sigma}^{-1/2}\boldsymbol{\mu}, \boldsymbol{I})$
- **几何意义与 R 函数**: 等高面为椭球 $(\boldsymbol{x}-\boldsymbol{\mu})'\boldsymbol{\Sigma}^{-1}(\boldsymbol{x}-\boldsymbol{\mu}) = c^2$；R `mvtnorm` 包提供 `dmvnorm`/`pmvnorm`/`qmvnorm`/`rmvnorm`

## 03 条件分布和独立性
### 条件分布 (定理 3.3 & 3.4)
- **分块条件分布**: $(\boldsymbol{X}^{(1)}|\boldsymbol{X}^{(2)}=\boldsymbol{x}^{(2)}) \sim N_q(\boldsymbol{\mu}_{1\cdot 2}, \boldsymbol{\Sigma}_{11\cdot 2})$
- **条件均值与条件协方差**: $\boldsymbol{\mu}_{1\cdot 2} = \boldsymbol{\mu}^{(1)} + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\boldsymbol{x}^{(2)}-\boldsymbol{\mu}^{(2)})$，$\boldsymbol{\Sigma}_{11\cdot 2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21} \le \boldsymbol{\Sigma}_{11}$
- **线性回归分解 (定理 3.4)**: $\boldsymbol{X}^{(1)}$ 与 $\boldsymbol{X}^{(2)} - \boldsymbol{\Sigma}_{21}\boldsymbol{\Sigma}_{11}^{-1}\boldsymbol{X}^{(1)}$ 相互独立

### 独立性判定
- **不相关等价于独立**: 对多元正态向量，分块不相关 ($\boldsymbol{\Sigma}_{ij}=\boldsymbol{0}$) $\Leftrightarrow$ 相互独立；$\boldsymbol{\Sigma}$ 为对角阵 $\Leftrightarrow$ 各分量相互独立
- **线性变换独立性**: $\boldsymbol{Y}=\boldsymbol{A}\boldsymbol{X}+\boldsymbol{a}$ 与 $\boldsymbol{Z}=\boldsymbol{B}\boldsymbol{X}+\boldsymbol{b}$ 独立 $\Leftrightarrow \boldsymbol{A}\boldsymbol{\Sigma}\boldsymbol{B}' = \boldsymbol{0}$

### 偏相关系数与全相关系数
- **偏相关系数**: 给定 $\boldsymbol{X}^{(2)}$ 下条件相关系数 $R_{11\cdot 2} = [\operatorname{diag}(\boldsymbol{\Sigma}_{11\cdot 2})]^{-1/2}\boldsymbol{\Sigma}_{11\cdot 2}[\operatorname{diag}(\boldsymbol{\Sigma}_{11\cdot 2})]^{-1/2}$
- **全相关系数 ($R$)**: $R = \left(\boldsymbol{\Sigma}_{y\boldsymbol{X}}\boldsymbol{\Sigma}_{\boldsymbol{X}\boldsymbol{X}}^{-1}\boldsymbol{\Sigma}_{\boldsymbol{X}y}/\sigma_{yy}\right)^{1/2} = \max \operatorname{corr}(y, \boldsymbol{a}'\boldsymbol{X})$

### 矩阵多元正态分布
- **定义与定理 3.5**: 随机样本矩阵 $\boldsymbol{X}_{n\times p} \sim N_{n\times p}(\boldsymbol{1}_n\boldsymbol{\mu}', \boldsymbol{I}_n\otimes\boldsymbol{\Sigma}) \Leftrightarrow \operatorname{Vec}(\boldsymbol{X}') \sim N_{np}(\boldsymbol{1}_n\otimes\boldsymbol{\mu}, \boldsymbol{I}_n\otimes\boldsymbol{\Sigma})$
- **双线性变换 (3.3.12)**: $\boldsymbol{Z} = \boldsymbol{A}\boldsymbol{X}\boldsymbol{B}+\boldsymbol{D} \sim N_{k\times q}(\boldsymbol{A}\boldsymbol{M}\boldsymbol{B}+\boldsymbol{D}, (\boldsymbol{A}\boldsymbol{A}')\otimes(\boldsymbol{B}'\boldsymbol{\Sigma}\boldsymbol{B}))$

## 04 参数的极大似然估计及性质
### 样本统计量
- **样本均值 ($\bar{\boldsymbol{X}}$)**: $\bar{\boldsymbol{X}} = \frac{1}{n}\sum_{i=1}^n \boldsymbol{X}_{(i)} = \frac{1}{n}\boldsymbol{X}'\boldsymbol{1}_n$
- **样本离差阵 ($\boldsymbol{A}$)**: $\boldsymbol{A} = \sum_{i=1}^n (\boldsymbol{X}_{(i)}-\bar{\boldsymbol{X}})(\boldsymbol{X}_{(i)}-\bar{\boldsymbol{X}})' = \boldsymbol{X}'(\boldsymbol{I}_n - \frac{1}{n}\boldsymbol{1}_n\boldsymbol{1}_n')\boldsymbol{X}$
- **样本协方差阵与相关阵**: $\boldsymbol{S} = \frac{1}{n-1}\boldsymbol{A}$，$\widetilde{\boldsymbol{R}} = \widetilde{\boldsymbol{D}}^{-1/2}\boldsymbol{S}\widetilde{\boldsymbol{D}}^{-1/2}$

### MLE 与抽样分布
- **MLE 与矩估计 (定理 3.6)**: $\widehat{\boldsymbol{\mu}} = \bar{\boldsymbol{X}}$，$\widehat{\boldsymbol{\Sigma}} = \frac{1}{n}\boldsymbol{A}$（$n > p$）
- **抽样分布 (定理 3.8)**: $\bar{\boldsymbol{X}} \sim N_p(\boldsymbol{\mu}, \frac{1}{n}\boldsymbol{\Sigma})$，$\boldsymbol{A} \sim W_p(n-1, \boldsymbol{\Sigma})$（Wishart 分布），且 $\bar{\boldsymbol{X}}$ 与 $\boldsymbol{A}$ 相互独立；$P(\boldsymbol{A}>\boldsymbol{0})=1 \Leftrightarrow n>p$

### 估计量评价准则
- **无偏性与有效性**: $(\bar{\boldsymbol{X}}, \boldsymbol{S})$ 是 $(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ 的无偏估计与有效估计
- **相合性与充分性**: $(\bar{\boldsymbol{X}}, \frac{1}{n}\boldsymbol{A})$ 是强相合估计；$(\bar{\boldsymbol{X}}, \boldsymbol{S})$ 是 $(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ 的充分统计量

## 05 多元正态分布的假设检验问题
### 一元正态性评估
- **图形与区间检验**: $1\sigma/2\sigma/3\sigma$ 经验比例直方图、正态 Q-Q 图（$n \ge 20$）
- **量化检验**: 偏度 $Skw$、峰度 $Kurt$、Shapiro-Wilks 检验 ($W$)、Anderson-Darling ($A^2$)、Kolmogorov-Smirnov ($D$)、Cramer-von Mises ($T$)

### 多元正态性评估
- **卡方 Q-Q 图**: 检验马氏距离平方 $\widehat{d_i^2} = (\boldsymbol{x}_i-\bar{\boldsymbol{x}})'\boldsymbol{S}^{-1}(\boldsymbol{x}_i-\bar{\boldsymbol{x}})$ 是否近似服从 $\chi_p^2$（$n-p \ge 25$）
- **Energy Test**: 基于 Energy distance $\varepsilon(X,Y) = 2E\|X-Y\| - E\|X-X'\| - E\|Y-Y'\| \ge 0$，$\varepsilon$ 过大拒绝正态性
