# Ch3 内生性和工具变量

## 第一节 大样本理论下的一致性

### 1.1 收敛概念与评价标准
- **依概率收敛**: $\lim_{n\to\infty} P(|x_n - a| > \varepsilon) = 0$，记为 $x_n \xrightarrow{p} a$（一致性：$\text{plim}\,\hat{\theta}_n = \theta$）
- **依均方收敛**: $\mu_n \to \mu$ 且 $\sigma_n^2 \to 0$，必推出依概率收敛
- **依分布收敛**: 累积分布函数收敛 $F_n(x) \to F(x)$，记为 $x_n \xrightarrow{d} x$（渐近无偏：$\lim E(\hat{\theta}_n) = \theta$）
- **大样本标准**: 重点关注一致性、渐近无偏性与渐近有效性（Granger 准则：一致性是最起码要求）

### 1.2 大样本 OLS 性质
- **小样本 vs 大样本**: 小样本无偏需严格外生性 $E(\varepsilon \mid X) = 0$（AR(1) 等动态模型常违背）；大样本一致只需前定解释变量 $E(x_{ik}\varepsilon_i) = 0$
- **渐近正态性**: 在线性、遍历平稳、前定、秩条件及鞅差分序列 (MDS) 假设下，$\sqrt{n}(\hat{\beta} - \beta) \xrightarrow{d} N(0, [E(x_i x_i')]^{-1} S [E(x_i x_i')]^{-1})$，天然允许条件异方差

## 第二节 内生性

### 2.1 内生性定义与三大成因
- **核心定义**: 解释变量与扰动项相关 $E(\varepsilon_t X_t) \neq 0$，导致 OLS 不一致、渐近有偏、常规检验失效
- **1. 测量误差**: 解释变量测量误差 $\gamma_t$ 造成衰减偏误（attenuation bias），$\hat{\beta}_1 \xrightarrow{p} \beta_1 \frac{1}{1 + \sigma_\gamma^2/\sigma_{X^*}^2}$；被解释变量测量误差 $\omega_t$ 不致偏但增大渐近方差
- **2. 模型误设**: 遗漏与解释变量相关的重要变量、函数形式误设、以及样本选择偏差与自选择问题
- **3. 联立性偏误**: 双向因果关系导致解释变量与扰动项联立决定（如奖金与产出、收入与消费）

### 2.2 内生性解决方案与自然实验
- **常用方法**: 自然实验、工具变量 (IV)、双重差分 (DID)、动态面板数据模型、断点回归 (RDD)、Heckman 两步法、PSM
- **两类自然实验**: 第一类完全随机决定分组可直接 OLS/DID（如 Card & Krueger 1994 最低工资）；第二类仅部分决定分组需作为 IV（如 Angrist 1990 越战抽签、Miguel et al. 2004 降雨量）

## 第三节 工具变量（IV）与 2SLS

### 3.1 IV 的两大核心条件
- **相关性条件**: 工具变量 $Z_t$ 与内生解释变量 $X_t$ 高度相关，$Q_{ZX} = E(Z_t X_t')$ 有限且满秩
- **外生性条件（排他性约束）**: $E(\varepsilon_t \mid Z_t) = 0$ 或 $E(Z_t \varepsilon_t) = 0$，$Z_t$ 只能通过 $X_t$ 唯一影响 $Y_t$
- **滞后项警示**: 内生变量滞后项 $X_{t-1}$ 作 IV 存在相关性与无序列相关外生性之间的内在矛盾，通常应避免
- **经典 IV 案例**: 河流数目（学区竞争）、降雨量（看电视与自闭症、内战）、地理与人口中心距离（腐败）、出生季度（教育收益率）

### 3.2 两阶段最小二乘法（2SLS）与 GMM
- **两步思想**: 第一阶段 $\hat{X} = Z(Z'Z)^{-1}Z'X$ 分离外生投影，第二阶段 $Y$ 对 $\hat{X}$ 回归：$\hat{\beta}_{2SLS} = [X'Z(Z'Z)^{-1}Z'X]^{-1}[X'Z(Z'Z)^{-1}Z']Y$
- **矩估计视角**: 正交条件 $E(z_i \varepsilon_i) = 0$，恰好识别（$l=K$）时 $\hat{\beta}_{IV} = (Z'X)^{-1}Z'y$；过度识别（$l>K$）用 2SLS
- **GMM vs 2SLS**: 球形扰动下 2SLS 最有效；存在异方差或自相关时，GMM 比 2SLS 更有效（GMM 之于 2SLS 类似 GLS 之于 OLS）

## 第四节 工具变量相关检验与 Stata 操作

### 4.1 四大核心检验
- **不可识别检验**: 检验秩条件是否满秩，$H_0$: 不可识别；同方差用 Anderson LM，异方差用 Kleibergen-Paap rk LM（`ivreg2`）
- **弱工具变量检验**: 一阶段偏 $R^2$（Shea's partial $R^2$）、一阶段 $F > 10$（`estat firststage`）、Cragg-Donald Wald F / Kleibergen-Paap Wald rk F（对照 Stock-Yogo 临界值）；弱 IV 可改用 `ivregress liml` 或剔除冗余 IV（`redundant()`）
- **过度识别检验**: 仅在 $l > K$ 时可检验，$H_0$: 所有 IV 外生；统计量有 Sargan（`estat overid`）、Hansen J 与检验子集外生性的 C 统计量（`ivreg2 ..., orthog()`）
- **内生性检验**: 传统 Hausman 检验（同方差假设，异方差时可用 Bootstrap）与稳健的 Durbin-Wu-Hausman 检验（`estat endogenous`）及面板 D-M 检验（`dmexogxt`）

### 4.2 Stata 实操命令
- **核心命令**: 严禁手动两步 OLS（标准误错误），应使用官方 `ivregress 2sls y x1 (x2 = z1 z2), r first` 或外部 `ivreg2` / `xtivreg2`
- **经典案例 (`grilic.dta`)**: 用 `(med, kww)` 替代含测量误差与外生性争议的 `(mrt, age)` 作为 `iq` 的 IV，通过过度识别与弱工具检验得到合理教育回报率
