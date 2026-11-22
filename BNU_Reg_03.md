# 第3章 多元线性回归

## 3.1 多元线性回归模型
- **模型形式**: $Y = X\beta + \epsilon$，设计矩阵 $X_{n \times (p+1)}$ 需满足列满秩 $rk(X) = p + 1 < n$
- **基本假定**: 解释变量确定性；Gauss-Markov 条件 $E(\epsilon)=0, Var(\epsilon)=\sigma^2 I_n$；正态假定 $\epsilon \sim N(0, \sigma^2 I_n)$

## 3.2 参数估计与性质
### 最小二乘估计 (LSE) 与帽子矩阵
- **正规方程组**: $X'X\hat{\beta} = X'Y \implies \hat{\beta} = (X'X)^{-1}X'Y$
- **拟合值与残差**: $\hat{Y} = HY$，$e = (I_n - H)Y$，帽子矩阵 $H = X(X'X)^{-1}X'$
- **帽子矩阵性质**: 对称幂等 ($H=H'=H^2$)，$rk(H)=tr(H)=p+1$，$rk(I_n-H)=n-p-1$
- **$\sigma^2$ 无偏估计**: $\hat{\sigma}^2 = \frac{e'e}{n-p-1} = \frac{Y'(I_n-H)Y}{n-p-1}$（MLE 为 $\frac{e'e}{n}$）
### 性质与分布理论
- **均值与协方差**: $E(\hat{\beta}) = \beta$，$Var(\hat{\beta}) = \sigma^2(X'X)^{-1}$；中心化时截距与斜率不相关
- **Gauss-Markov 定理**: $c'\hat{\beta}$ 为 BLUE；正态假定下为 UMVUE
- **正态抽样分布**: $\hat{\beta} \sim N(\beta, \sigma^2(X'X)^{-1})$，$\frac{SSE}{\sigma^2} \sim \chi^2(n-p-1)$，且 $\hat{\sigma}^2 \perp \hat{\beta}$
- **残差性质**: $E(e)=0$，$Var(e)=\sigma^2(I_n-H)$，$Cov(e, \hat{Y})=Cov(e, \hat{\beta})=0$，$X'e=0$

## 3.3 显著性检验、拟合优度与推断
### 三类显著性检验
- **整体 $F$ 检验**: $H_0: \beta_1 = \dots = \beta_p = 0$，$F = \frac{SSR/p}{SSE/(n-p-1)} \sim F(p, n-p-1)$
- **单系数 $t$ 检验 / 偏 $F$ 检验**: $t_j = \frac{\hat{\beta}_j}{\sqrt{\hat{\sigma}^2 c_{jj}}} \sim t(n-p-1)$，等价于 $F_j = t_j^2 = \frac{\Delta SSR_{(-j)}}{SSE/(n-p-1)}$
- **部分系数 $F$ 检验**: $H_0: \beta_B = 0$，$F = \frac{(SSE_A - SSE)/p_0}{SSE/(n-p-1)} \sim F(p_0, n-p-1)$
### 拟合优度与区间预测
- **决定系数**: $R^2 = \frac{SSR}{SST}$；调整决定系数 $R_a^2 = 1 - \frac{n-1}{n-p-1}(1-R^2)$
- **置信域（椭球）**: $\frac{(\hat{\beta}-\beta)'X'X(\hat{\beta}-\beta)}{(p+1)\hat{\sigma}^2} \le F_\alpha(p+1, n-p-1)$
- **区间预测**: $y_0$ 预测区间 $\hat{y}_0 \pm t_{\alpha/2}(n-p-1)\sqrt{\hat{\sigma}^2(1 + x_0'(X'X)^{-1}x_0)}$

## 3.4 回归系数的解释与 3.5 偏相关系数
- **常规解释**: 固定其他自变量，$x_j$ 增 1 单位时 $E(y)$ 的变化量
- **偏回归图 (附加变量图)**: $e_{y|x_{(-j)}}$ 对 $e_{x_j|x_{(-j)}}$ 回归，过原点且斜率为 $\hat{\beta}_j$
- **正交投影分解**: $H = H_{(-j)} + H_{Z_j}$，其中 $Z_j = (I_n - H_{(-j)})X_j$
- **偏相关系数**: 消除其余变量影响后的纯相关性，$r_{12;3,\dots,p} = \frac{-\Delta_{12}}{\sqrt{\Delta_{11}\Delta_{22}}} = Cor(e_{x_1|x_{(-1)}}, e_{y|x_{(-1)}})$
- **偏决定系数**: $r_{y1;2,\dots,p}^2 = \frac{SSE_{(-1)} - SSE}{SSE_{(-1)}} = \frac{F_1}{F_1 + n - p - 1}$

## 3.6 中心化与标准化
- **中心化**: 平移变换不改变斜率估计与 $R^2$，截距变为 $\bar{y}$（$Y$ 也中心化则为 $0$），可降参数维数与舍入误差
- **标准化**: $\hat{\tilde{\eta}} = \frac{1}{sd(y)}D_X \hat{\tilde{\beta}}$（标准化回归系数），不改变相关阵、$F$、斜率 $t$ 与 $R^2$

## 3.7 综合案例（民航客运量）
- **建模五步**: 设变量收数据 $\to$ EDA (`pairs`, `corrplot`) $\to$ `lm()` 拟合 $\to$ 回归诊断 $\to$ 模型应用
- **共线性信号**: 案例中 $R^2=0.9959, F=686.4$ 极显著，但单变量 $t$ 检验均不显著且部分系数为负
