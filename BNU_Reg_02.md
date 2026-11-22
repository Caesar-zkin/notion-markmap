# 第2章 一元线性回归

## 2.1 一元线性回归模型
### 理论模型与样本模型
- **理论模型**: $y = \beta_0 + \beta_1 x + \epsilon$，其中 $E(\epsilon|x) = 0, Var(\epsilon|x) = \sigma^2$（等价于 $E(y|x) = \beta_0 + \beta_1 x$）
- **损失视角**: 平方损失下最优预测为条件期望 $E(y|x)$；绝对值损失下为条件中位数 $\text{Median}(y|x)$
- **参数含义**: $\beta_0 = E(y|x=0)$ 为截距；$\beta_1$ 为 $x$ 每增 1 单位时 $E(y)$ 的变化量
- **Gauss-Markov 条件**: $E(\epsilon_i) = 0$，$Var(\epsilon_i) = \sigma^2$，$Cov(\epsilon_i, \epsilon_j) = 0$ ($i \neq j$)
- **矩阵表示**: $Y = X\beta + \epsilon$，$E(\epsilon) = 0$，$Var(\epsilon) = \sigma^2 I_n$

## 2.2 参数估计及其性质
### 最小二乘估计 (LSE)
- **目标函数**: 极小化离差平方和 $Q(\beta_0, \beta_1) = \sum_{i=1}^n (y_i - \beta_0 - \beta_1 x_i)^2$
- **正规方程组**: $\sum e_i = 0$ 且 $\sum x_i e_i = 0$（经验回归线必过重心 $(\bar{x}, \bar{y})$）
- **估计公式**: $\hat{\beta}_1 = \frac{L_{xy}}{L_{xx}}$，$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$
- **几何解释**: $\hat{\beta}_1 \tilde{X}_c$ 是中心化 $Y_c$ 在 $\tilde{X}_c$ 上的投影；$\cos\theta = Cor(\tilde{X}, Y)$
- **$\sigma^2$ 无偏估计**: $\hat{\sigma}^2 = \frac{1}{n-2}\sum_{i=1}^n e_i^2 = \frac{SSE}{n-2}$
### 极大似然估计 (MLE)
- **正态假设**: $\epsilon_i \stackrel{\text{i.i.d.}}{\sim} N(0, \sigma^2)$ 时，$\hat{\beta}_{MLE} = \hat{\beta}_{LSE}$
- **方差估计**: $\hat{\sigma}^2_{MLE} = \frac{1}{n}\sum_{i=1}^n e_i^2$（有偏）
### LSE 的统计性质
- **线性与无偏性**: $\hat{\beta}_0, \hat{\beta}_1$ 为 $y_i$ 的线性组合，且 $E(\hat{\beta}_0)=\beta_0, E(\hat{\beta}_1)=\beta_1$
- **方差与协方差**: $Var(\hat{\beta}_1) = \frac{\sigma^2}{L_{xx}}$，$Var(\hat{\beta}_0) = (\frac{1}{n} + \frac{\bar{x}^2}{L_{xx}})\sigma^2$，$Cov(\hat{\beta}_0, \hat{\beta}_1) = -\frac{\bar{x}}{L_{xx}}\sigma^2$
- **Gauss-Markov 定理**: 在 Gauss-Markov 条件下，$\hat{\beta}_0, \hat{\beta}_1$ 是最佳线性无偏估计 (BLUE)
- **正态分布**: $\hat{\beta}_1 \sim N(\beta_1, \frac{\sigma^2}{L_{xx}})$，$\hat{\beta}_0 \sim N(\beta_0, (\frac{1}{n}+\frac{\bar{x}^2}{L_{xx}})\sigma^2)$

## 2.3 区间估计与 2.4 显著性检验
### 区间估计
- **$\beta_1$ 的 $1-\alpha$ 置信区间**: $\hat{\beta}_1 \pm t_{\alpha/2}(n-2)\sqrt{\hat{\sigma}^2 / L_{xx}}$
### 假设检验
- **$t$ 检验**: $H_0: \beta_1 = 0$，统计量 $t = \frac{\hat{\beta}_1}{\sqrt{\hat{\sigma}^2 / L_{xx}}} \sim t(n-2)$
- **$F$ 检验与方差分解**: $SST = SSR + SSE$，统计量 $F = \frac{SSR/1}{SSE/(n-2)} \sim F(1, n-2)$，且 $F = t^2$
- **决定系数**: $R^2 = \frac{SSR}{SST} = r^2 = \frac{F}{F + n - 2} \in [0, 1]$

## 2.5 残差及其性质
- **基本性质**: $E(e_i) = 0$，$\sum e_i = 0$，$\sum x_i e_i = 0$，$Cov(e_i, \hat{y}_j) = Cov(e_i, \hat{\beta}_k) = 0$
- **方差与杠杆值**: $Var(e_i) = (1 - h_{ii})\sigma^2$，其中杠杆值 $h_{ii} = \frac{1}{n} + \frac{(x_i - \bar{x})^2}{L_{xx}}$
- **高杠杆点**: $x_i$ 远离 $\bar{x}$ 时 $h_{ii} \to 1$，$Var(e_i) \to 0$，该点会将拟合直线拉向自身

## 2.6 预测和控制
- **单值预测与 MSEP**: $\hat{y}_0 = \hat{\beta}_0 + \hat{\beta}_1 x_0$，$MSEP(\hat{y}_0) = (1 + h_{00})\sigma^2$
- **$y_0$ 的 $1-\alpha$ 预测区间**: $\hat{y}_0 \pm t_{\alpha/2}(n-2)\sqrt{\hat{\sigma}^2(1 + h_{00})}$
- **$E(y_0)$ 的 $1-\alpha$ 置信区间**: $\hat{y}_0 \pm t_{\alpha/2}(n-2)\sqrt{\hat{\sigma}^2 h_{00}}$
- **提高精度**: 增大样本量 $n$、分散采集 $x$（增大 $L_{xx}$）、$x_0$ 靠近 $\bar{x}$
- **控制问题**: 在因果前提下反解 $x$ 使 $P(T_1 < y < T_2) = 1-\alpha$（要求 $T_2 - T_1 > 4\hat{\sigma}$）
