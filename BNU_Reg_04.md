# 第4章 回归诊断

## 诊断框架与 4.1 残差分析
### 诊断两个维度
- **数据本身**: 异常点（$Y$ 空间）、高杠杆点（$X$ 空间）、强影响点、多重共线性（$rk(X) < p+1$）
- **误差项**: 异方差性、自相关性、正态性违背
### 四种残差与杠杆值
- **杠杆值**: $h_{ii} = \frac{1}{n} + (\tilde{x}_i - \bar{X})'(\tilde{X}_c'\tilde{X}_c)^{-1}(\tilde{x}_i - \bar{X}) \in [0, 1]$，均值 $\bar{h} = \frac{p+1}{n}$；$h_{ii} > 2\bar{h}$ 为高杠杆点
- **普通残差**: $e_i = y_i - \hat{y}_i$，$Var(e_i) = (1 - h_{ii})\sigma^2$
- **内学生化残差**: $r_i = \frac{e_i}{\sqrt{(1 - h_{ii})\hat{\sigma}^2}}$（近似标准正态）
- **预测/删除残差**: $e_{(i)} = y_i - \hat{y}_{(i)} = \frac{e_i}{1 - h_{ii}}$，$Var(e_{(i)}) = \frac{\sigma^2}{1 - h_{ii}}$
- **外学生化残差**: $r_{(i)} = \frac{e_i}{\sqrt{(1 - h_{ii})\hat{\sigma}_{(-i)}^2}} = r_i\sqrt{\frac{n-p-2}{n-p-1-r_i^2}} \sim t(n-p-2)$
- **残差图**: 95% 落于 $[-2,2]$ 为正常；喇叭形提示异方差；曲线提示非线性/自相关/漏变量

## 4.2 异常点、高杠杆点和强影响点
### 异常点诊断
- **删除学生化残差**: $|r_{(i)}| > 3$ 判定为异常点（避免 $r_i$ 的掩盖）
- **均值漂移模型**: $Y = X\beta + \gamma d_i + \epsilon$，$H_0: \gamma=0$ 的统计量 $F_i = r_{(i)}^2 \sim F(1, n-p-2)$
- **多异常点效应**: 掩盖效应 (masking，漏检) 与淹没效应 (swamping，误判)
### 强影响点诊断
- **Cook 距离**: $D_i = \frac{1}{p+1}\frac{h_{ii}}{1-h_{ii}}r_i^2$；$D_i > 1$（或 $D_i > F_{0.5}(p+1, n-p-1)$）为强影响点
- **Welsch-Kuh (DFFITS)**: $WK_i = \sqrt{\frac{h_{ii}}{1-h_{ii}}r_{(i)}^2}$，准则 $|WK_i| > 2\sqrt{\frac{p+1}{n-p-1}}$
- **Hadi 统计量**: $H_i = \frac{h_{ii}}{1-h_{ii}} + \frac{p+1}{1-h_{ii}}\frac{d_i^2}{1-d_i^2}$（杠杆与残差的**和**函数）

## 4.3 异方差性问题及其处理
- **后果**: OLS 仍无偏但不再是 BLUE；低估真实方差导致高估 $|t|$ 值（伪显著）
- **诊断**: 残差图 (`car::spreadLevelPlot`)、Spearman 等级相关检验 $r_s = 1 - \frac{6\sum d_i^2}{n(n^2-1)}$
- **加权最小二乘法 (WLS)**: $\hat{\beta}^{WLS} = (X'WX)^{-1}X'WY$，最优权 $w_i = 1/\sigma_i^2$（常用 $w_i = 1/x_{ij}^m$）
- **Box-Cox 变换**: $y^{(\lambda)} = (y^\lambda - 1)/\lambda$ ($\lambda \neq 0$) 或 $\ln y$ ($\lambda=0$)，MLE 求最优 $\hat{\lambda}$

## 4.4 自相关性问题及其处理
- **后果**: 与异方差相同（非 BLUE、低估方差、伪显著），常见于时间序列
- **诊断**: $(e_{t-1}, e_t)$ 散点图、序列图、一阶自相关系数 $\hat{\rho}$
- **D.W. 检验**: $D.W. = \frac{\sum_{t=2}^n (e_t - e_{t-1})^2}{\sum_{t=1}^n e_t^2} \approx 2(1 - \hat{\rho}) \in [0, 4]$，对照 $(d_L, d_U)$ 判定（存在两个不确定区）
- **处理**: Cochrane-Orcutt 迭代法、广义差分法

## 4.5 多重共线性问题及其处理
- **表现与后果**: $|X'X| \approx 0$；方差急剧膨胀、$F$ 高而 $t$ 不显著、系数符号反常
### 诊断指标
- **Klein 判别法**: 两自变量 $|r_{ij}| > R$
- **条件数**: $\kappa_i = \sqrt{\lambda_1 / \lambda_i}$；$\kappa_i \in [10, 100)$ 中等较强，$\kappa_i \ge 100$ 严重
- **方差膨胀因子 (VIF)**: $VIF_i = \frac{1}{1 - R_i^2} = (\mathbf{r}^{-1})_{ii}$，$VIF_i \ge 10$ 存在强共线性
### 处理与岭回归
- **数据处理法**: 剔除变量、增加样本、合并相关变量
- **岭回归**: $\hat{\beta}(\lambda) = (X'X + \lambda I_p)^{-1}X'Y$（不压缩截距），以有偏换取 $MSE(\hat{\beta}(\lambda)) < MSE(\hat{\beta})$
- **SVD 视角**: $\hat{Y}_\lambda = \sum \frac{d_i^2}{d_i^2 + \lambda}\langle u_i, Y\rangle u_i$，对小奇异值方向施加更强压缩
- **$\lambda$ 选取**: 岭迹法（系数稳定、符号合理）、VIF 法（使所有岭 VIF $\le 10$）
