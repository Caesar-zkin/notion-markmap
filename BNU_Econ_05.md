# Ch05 面板数据模型的分析II：扩展研究

## 第一节 面板工具变量法

### 1.1 面板 IV 与 GMM
- **两步处理**: 先通过 FE 离差或 FD 差分消去个体效应 $u_i$，再对变换后方程做 2SLS（`xtivreg ..., fe` / `fd` / `re`）
- **过度识别与 GMM**: 外部命令 `xtivreg2 ..., fe gmm` 支持面板 GMM 与弱工具/不可识别检验；`xtoverid` 可检验过度识别
- **面板 IV 维度要求**: 工具变量必须同时具备时间与横截面二维变动（如“地形坡度 $\times$ 年份”或“美国小麦产量 $\times$ 受援概率”），避免与单维固定效应共线

### 1.2 Hausman-Taylor 估计量
- **解决痛点**: 估计不随时间变化的内生变量 $z_{2i}$ 的系数 $\delta_2$
- **工具变量构造**: 外生变量 $(x_{1,it}, z_{1i})$ 自身作 IV；时变内生变量 $x_{2,it}$ 用离差 $(x_{2,it} - \bar{x}_{2,i})$ 作 IV；非时变内生变量 $z_{2i}$ 用 $\bar{x}_{1,i}$ 作 IV（要求 $x_1$ 个数 $\ge z_2$ 个数）
- **Stata 命令**: `xthtaylor depvar indepvars, endog(varlist) amacurdy`

### 1.3 Bartik 工具变量（Shift-Share IV）
- **构造原理**: 固定基期地区行业份额（Share）与全国行业增长率/冲击（Shock）的内积 $B_{it} = \sum_{k=1}^K Z_{iks} g_{kt}$
- **识别条件**: 满足基期份额 $Z_{iks}$ 外生性（Goldsmith-Pinkham et al., 2020）或外部冲击 $g_{kt}$ 准随机外生性（Borusyak et al., 2022），配合平衡性检验与 `ssaggregate` 聚类标准误

## 第二节 动态面板数据模型

### 2.1 动态面板偏差
- **模型形式**: $y_{it} = \alpha + \rho y_{i,t-1} + x_{it}'\beta + z_i'\delta + u_i + \varepsilon_{it}$
- **OLS 与 FE/FD 不一致**: $y_{i,t-1}$ 与 $u_i$ 相关，组内均值 $\overline{Ly}_i$ 与 $\bar{\varepsilon}_i$ 相关，$\Delta y_{i,t-1}$ 与 $\Delta \varepsilon_{it}$ 相关，导致短面板中 OLS/FE/FD 均不一致

### 2.2 差分 GMM（Difference GMM）
- **Anderson-Hsiao 估计量**: 对一阶差分方程 $\Delta y_{it} = \rho \Delta y_{i,t-1} + \Delta x_{it}'\beta + \Delta\varepsilon_{it}$ 用 $y_{i,t-2}$ 作 IV 进行 2SLS
- **Arellano-Bond 估计量**: 利用所有更高阶滞后 $\{y_{i,t-2}, y_{i,t-3}, \dots\}$ 作为 GMM 式展开工具变量
- **四大局限**: 差分下前定变量变成内生（需用更高阶滞后作 IV）、$T$ 较大时 IV 过多导致弱 IV 与 Hansen 检验虚高、无法估计非时变变量 $z_i$、序列高度持续（接近随机游走）时滞后项为弱 IV

### 2.3 水平 GMM 与系统 GMM
- **水平 GMM (Arellano-Bover)**: 用 $\{\Delta y_{i,t-1}, \Delta y_{i,t-2}, \dots\}$ 作水平方程的 IV，可估计 $z_i$，但要求收敛速度与 $u_i$ 不相关（序列处于稳态附近）
- **系统 GMM (Blundell-Bond)**: 将差分方程与水平方程联立进行 GMM 估计，提高估计效率并可估计 $z_i$

### 2.4 两大核心检验与 Stata 操作
- **扰动项自相关检验 (`estat abond`)**: 原假设 $\{\varepsilon_{it}\}$ 无自相关，等价于一阶差分扰动存在 AR(1) 且不存在 AR(2)（$p > 0.05$）
- **过度识别检验**: `estat sargan`（同方差假设）或 `xtabond2` 提供的稳健 Hansen J 检验
- **Stata 命令**: 官方 `xtabond`（差分 GMM）、`xtdpdsys`（系统 GMM）与外部命令 `xtabond2`（支持 `gmm()`、`iv()`、`nolevel`、`twostep robust`）

## 第三节 长面板（$T$ 大 $n$ 小）
- **扰动项三大问题**: 组间异方差（`xttest3` 检验）、组内自相关（`xtserial` 检验）、组间同期/截面相关（`xttest2` 或 `xtcsd` 检验）
- **两类估计策略**: OLS + 面板校正标准误（`xtpcse`）或仅纠正 AR(1) 的 Prais-Winsten 估计（`corr(ar1)` / `corr(psar1)`）；全面 FGLS / 迭代 FGLS（`xtgls ..., panels() corr() igls`）
- **变系数与平稳性**: 固定系数系统 SUR、随机系数模型（`xtrc`）；面板单位根检验（`xtunitroot`）与协整检验（`xtcointtest`）

## 第四节 重复截面数据与组群分析
- **重复截面数据 (TSICS)**: 不同时期抽样个体不同；RE 可直接混合 OLS，FE 无法做组内或一阶差分
- **组群分析 / 伪面板 (Cohort Analysis)**: 按固定特征（如出生年份）分组求组群时点均值 $y_{ct} = x_{ct}'\beta + u_c + \varepsilon_{ct}$ 后做 FE，需注意年龄-时期-组群（APC）共线性识别问题
