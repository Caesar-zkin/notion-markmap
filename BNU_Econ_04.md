# Ch04 面板数据模型的分析 I：基础

## 第一节 面板数据模型简介

### 1.1 基本概念与数据特征
- **面板数据 (Panel Data)**: 同时包含横截面（$i=1,\dots,n$）与时间序列（$t=1,\dots,T$）二维信息
- **分类**: 短面板（$n$ 大 $T$ 小）与长面板（$T$ 大 $n$ 小）；静态面板与动态面板（含 $y_{i,t-1}$）；平衡面板与非平衡面板
- **三大优势**: 解决不随时间变化的遗漏变量（个体异质性）问题、提供个体动态行为信息、增大样本容量提高估计精度
- **主要局限**: 同一个体不同期扰动项常存在自相关（不满足 i.i.d.）、收集成本高

### 1.2 个体效应模型与策略选择
- **一般形式**: $y_{it} = x_{it}'\beta + z_i'\delta + u_i + \varepsilon_{it}$，其中 $x_{it}$ 随时间变化，$z_i$ 为可观测非时变特征，$u_i$ 为不可观测个体异质性
- **混合回归 (Pooled OLS)**: 假设全部 $u_i = 0$（相同截距与斜率），可用固定效应输出中的 $F$ 检验判断是否适用
- **固定效应 (FE) vs 随机效应 (RE)**: 若 $u_i$ 与 $(x_{it}, z_i)$ 中某个变量相关则为 FE（OLS 不一致）；若与全部解释变量均不相关则为 RE（OLS 一致但无效）
- **扩展模型**: 随机系数模型 $y_{it} = x_{it}'(\beta + h_i) + z_i'\delta + u_i + \varepsilon_{it}$ 与分层模型

## 第二节 固定效应模型的估计方法

### 2.1 三大估计方法
- **组内法 (Within Estimator)**: 对时间去均值 $\tilde{y}_{it} = \tilde{x}_{it}'\beta + \tilde{\varepsilon}_{it}$ 消去 $u_i$；缺点是同时消去 $z_i'\delta$，且要求严格外生性 $E(\varepsilon_{it} \mid x_{i1},\dots,x_{iT}) = 0$
- **最小二乘虚拟变量法 (LSDV)**: 加入 $n-1$ 个个体虚拟变量（`reg y x i.id`），与组内法斜率估计等价，但 $n$ 很大时损失大量自由度
- **一阶差分法 (FD)**: $\Delta y_{it} = \Delta x_{it}'\beta + \Delta \varepsilon_{it}$ 消去 $u_i$；当 $T>2$ 且 $\varepsilon_{it}$ 为 i.i.d. 时，FE 比 FD 更有效

### 2.2 时间效应与个体时间趋势
- **双向固定效应 (Two-way FE)**: 同时控制个体效应 $u_i$ 与时间效应 $\lambda_t$（`xtreg y x i.year, fe r`），或用线性时间趋势 $\gamma t$ 简化
- **随机趋势 / 随机增长模型**: 引入个体时间趋势 $\gamma_i t$（取对数后一阶差分 $\Delta y_{it} = \Delta x_{it}'\beta + \gamma_i + \Delta\varepsilon_{it}$）

## 第三节 随机效应模型的估计方法

### 3.1 复合扰动项结构与 OLS/FE
- **自相关结构**: 同一个体不同期复合扰动项相关系数 $\rho = \frac{\sigma_u^2}{\sigma_u^2 + \sigma_\varepsilon^2}$，协方差矩阵 $\Omega$ 为块对角阵
- **OLS 与 FE 表现**: RE 下混合 OLS 一致但非 BLUE（常规 $t/F$ 检验失效）；FE 依然一致但损失非时变变量 $z_i$ 信息

### 3.2 FGLS、组间估计与 LM 检验
- **可行广义最小二乘法 (FGLS)**: 对准离差模型 $y_{it} - \hat{\theta}\bar{y}_i = (x_{it} - \hat{\theta}\bar{x}_i)'\beta + (1-\hat{\theta})z_i'\delta + \text{error}$ 做 OLS，其中 $\theta = 1 - \frac{\sigma_\varepsilon}{(T\sigma_u^2 + \sigma_\varepsilon^2)^{1/2}}$，可同时估计 $\delta$
- **组间估计量 (BE)**: 对个体时间均值 $\bar{y}_i$ 回归，FE 下不一致，RE 下损失样本信息较少使用
- **Breusch-Pagan LM 检验**: 检验 $H_0: \sigma_u^2 = 0$（是否退化为混合 OLS），Stata 命令为 `xttest0`
- **双向随机效应**: 加入时间虚拟变量 `xtreg y x i.year, re r`

## 第四节 模型设定的若干问题

### 4.1 拟合优度与非平衡面板
- **三种 $R^2$**: 整体 $R^2$（overall）、组内 $R^2$（within，FE 推荐汇报）、组间 $R^2$（between）
- **非平衡面板**: 不影响组内法与 FGLS（$\theta_i$ 按 $T_i$ 调整），但需警惕内生性样本流失导致的估计不一致

### 4.2 聚类稳健标准误
- **三类标准误**: 普通标准误（同方差无自相关）、异方差稳健标准误（`robust`，要求观测独立）、聚类稳健标准误（`vce(cluster id)`，允许组内自相关）
- **短面板选择**: 同一横截面在不同时期常存在组内自相关，稳健标准误会低估真实标准误导致虚假显著，应使用聚类稳健标准误
- **聚类层级建议**: 至少聚类到核心解释变量所在层级（或县/市/省等更宏观层级），聚类数一般建议不少于 30（否则用 Bootstrap）

### 4.3 FE 与 RE 的选择：Hausman 检验
- **经典 Hausman 检验**: $H_0: u_i$ 与 $x_{it}, z_i$ 不相关（RE 一致且有效，FE 一致）；拒绝 $H_0$ 则选 FE
- **局限与改进**: 传统 Hausman 要求 RE 最有效（无异方差与自相关），否则需用 Bootstrap 或 Wooldridge (2010) 辅助回归检验；实证中因一致性考量普遍偏爱 FE

## 第五节 Stata 操作示例
- **面板设定与描述**: `xtset id year`、`xtdes`（结构）、`xtsum`（组内/组间/整体统计）、`xttab`、`xtline`
- **模型估计**: `xtreg y x i.year, fe r`（双向 FE + 聚类稳健标准误）、`xtreg y x, re` + `xttest0`（LM 检验）
