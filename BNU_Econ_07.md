# Ch7 双重差分模型II：扩展研究

## 第一节 横截面双重差分模型（队列 DID）

### 1.1 基本思想与适用场景
- **核心构造**: 当缺乏面板数据时，利用“地区暴露强度/政策冲击差异 $\times$ 出生队列（Cohort）差异”在横截面数据中构造双重差分（如程令国与张晔 2011 大饥荒与储蓄率、Chen et al. 2020 知青下乡与农村教育）
- **模型设定**: $y_{ics} = \beta_0 + \sum_c \alpha_c cohort_{ic} + \gamma_s drt_s + \sum_c \delta_c (cohort_{ic} \times drt_s) + X_{ics}\beta + \varepsilon_{ics}$
- **注意事项**: 需关注局部均衡分析的外推有效性（一般均衡效应与长期整体影响）

## 第二节 连续型双重差分模型

### 2.1 模型特征与优势
- **核心特征**: 将地区/个体维度的二分处理变量 $Treat_i$ 替换为连续变量（如茶叶种植面积、到地铁站距离、政府补助金额）
- **优势**: 保留更多数据变异信息，刻画不同地区或个体受政策影响的强度差异（如 Nancy Qian 2008、范子英等 2018）
- **两种用法**: 直接作为基准识别模型（$Continuous_i \times Post_t$），或在标准 DID 后作为稳健性检验

## 第三节 双重差分倾向得分匹配（PSM-DID）

### 3.1 原理与实施步骤
- **适用场景**: 存在不可观测且不随时间变化的组间差异（由 DID 消除）以及可观测协变量选择偏差（由 PSM 平衡）
- **前提假设**: 均值可忽略性假定 $E(y_{0t} - y_{0t'} \mid x, D=1) = E(y_{0t} - y_{0t'} \mid x, D=0)$ 与共同支撑假设
- **实施步骤**: 估计倾向得分 → 确定共同支撑域并匹配（最近邻/半径/核匹配）→ 平衡性检验 → 对前后差分 $(y_{1ti}-y_{0t'i})$ 与 $(y_{0tj}-y_{0t'j})$ 进行加权差分估计 $\widehat{ATT}$

## 第四节 多时点 DID（交错 DID）

### 4.1 TWFE 的偏误根源
- **模型差异**: 用随个体和时间变化的处理变量 $Treat_{it}$ 替代标准 $Treat_i \times Post_t$
- **Goodman-Bacon (2021) 分解**: 静态 TWFE 估计量是多个 $2\times 2$ DID 的加权平均，会将“早处理组”作为“晚处理组”的对照组（“坏的控制组”问题）
- **异质性处理效应偏误**: 当处理效应随组别或处理后时长异质时，负权重会导致 TWFE 估计有偏；动态情形下还会产生跨期交叉污染（Sun & Abraham, 2021）

### 4.2 异质性稳健估计量（三类方案）
- **1. 组别-时期平均处理效应 (CATT) 加权**: 仅用未处理/最后处理组作“好控制组”，命令有 `csdid` (Callaway & Sant'Anna)、`eventstudyinteract` (Sun & Abraham)、`did_multiplegt`、`twowayfeweights`
- **2. 插补估计量 (Imputation)**: 利用未处理观测估计反事实结果再作差，命令有 `did_imputation` / `event_plot` (Borusyak et al.)、`fect`、`did2s`
- **3. 堆叠回归 (Stacked Regression)**: 按相对事件时间给各处理组配对好控制组并堆叠回归，命令有 `stackedev` (Cengiz et al.)

### 4.3 多时点平行趋势与安慰剂检验
- **事件研究法**: $y_{it} = \alpha_i + \lambda_t + \sum_{l=-K}^{-2} \mu_l D_{it}^l + \sum_{l=0}^L \beta_l D_{it}^l + \epsilon_{it}$，需剔除一期（如 $l=-1$）作基期并对过早/过晚时期归并或截断
- **Stata 安慰剂包 (`didplacebo`)**: 支持时间安慰剂 `pbotime()`、空间安慰剂 `pbounit` 及混合安慰剂 `pbomix()`（陈强等, 2025）
