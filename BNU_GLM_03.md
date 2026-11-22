# 第3章 多分类广义线性模型

## 3.1 无序多分类模型与 3.2 多分类分布
- **哑变量编码**: $k$ 分类变量转为 $q = k - 1$ 维向量 $\mathbf{Y} = (1_{\{\text{类}1\}}(Y), \dots, 1_{\{\text{类}q\}}(Y))$，第 $k$ 类为参考类
- **一般形式**: $\mathbb{E}(\mathbf{Y}|\mathbf{X}) = \mathbf{h}(\mathbf{Z}\boldsymbol{\beta})$ 或 $\mathbf{g}(\mathbb{E}(\mathbf{Y}|\mathbf{X})) = \mathbf{Z}\boldsymbol{\beta}$
- **多分类逻辑回归**: $h_r(\mathbf{s}) = \frac{\exp(s_r)}{1 + \sum_{j=1}^q \exp(s_j)}$，等价于对数条件优势 $\log\frac{\mathbb{P}(Y=i|\mathbf{X})}{\mathbb{P}(Y=k|\mathbf{X})} = (1, \mathbf{X})\boldsymbol{\beta}_i$
- **分布性质**: $\mathbb{E}(\mathbf{Y}) = \boldsymbol{\pi}$，$\operatorname{var}(\mathbf{Y}) = \operatorname{diag}(\boldsymbol{\pi}) - \boldsymbol{\pi}^\top \boldsymbol{\pi}$

## 3.3 极大似然估计与 3.4 假设检验
- **得分与 Fisher 信息**: $\mathbf{s}(\boldsymbol{\beta}) = (\mathbf{Y} - \mathbf{h}(\boldsymbol{\beta}))\mathbf{V}(\boldsymbol{\beta})^{-1}\mathbf{D}(\boldsymbol{\beta})\mathbf{Z}$，$\mathbf{F}(\boldsymbol{\beta}) = \mathbf{Z}^\top \mathbf{D}^\top \mathbf{V}^{-1} \mathbf{D}\mathbf{Z}$
- **Logit 特例**: $\mathbf{D}(\boldsymbol{\beta}) = \mathbf{V}(\boldsymbol{\beta})$，故 $\mathbf{s}(\boldsymbol{\beta}) = (\mathbf{Y} - \mathbf{h}(\boldsymbol{\beta}))\mathbf{Z}$，$\mathbf{F}(\boldsymbol{\beta}) = \mathbf{Z}^\top \mathbf{V}(\boldsymbol{\beta})\mathbf{Z}$
- **参数检验 ($H_0: C\boldsymbol{\beta} = \xi$)**: 似然比 $\lambda$、Wald 统计量 $W$、得分统计量 $U$ 均渐近服从 $\chi^2(s)$；单分量 $Z_i \stackrel{a}{\sim} N(0,1)$
- **拟合优度**: 皮尔逊统计量 $\chi^2$ 与偏差统计量 $D = 2(l_0 - l(\hat{\boldsymbol{\beta}})) \stackrel{a}{\sim} \chi^2(g(k-1)-p)$

## 3.5 `vglm()` 与 3.6 选择模型
- **VGAM 实现**: `vglm(cbind(y1, y2, y3) ~ x1 + x2 + x3, family = multinomial(), data)`；`step4vglm()` 逐步回归
- **约束估计与检验**: 通过 `constraints = clist` 施加部分零约束或平行约束，并用 `lrtest()` 检验
- **选择模型**: 区分全局解释变量 $\mathbf{X}$（类别参数 $\boldsymbol{\beta}_r$）与类别解释变量 $\mathbf{W}$（全局参数 $\gamma$，如票价差 $W_i - W_k$）

## 3.7 多分类模型模拟研究
- **模拟流程**: `Matrix::bdiag` 构造分块对角设计矩阵 $\to$ `myH()` 算条件概率 $\to$ `sample(1:yNum, 1, prob=...)` 抽样 $\to$ `vglm()` 拟合验证

## 3.8 有序累积模型
- **潜变量机制**: $U = -\mathbf{X}\boldsymbol{\gamma} + \epsilon$，$Y=r \iff \theta_{r-1} < U \le \theta_r$，得 $\mathbb{P}(Y \le r|\mathbf{X}) = F(\theta_r + \mathbf{X}\boldsymbol{\gamma})$
- **三类常见链接**: 累积 Logit (比例优势模型)、累积 Probit、累积 Cloglog (比例风险模型)
- **R 实现**: `vglm(y ~ ..., family = cumulative(link="...", parallel=TRUE))`（`y` 需为有序因子 `ordered=TRUE`）
- **交互项解释**: 需计算主效应与交互效应之和（组合效应），并可反推参考类交互效应
- **部分比例累积模型**: 放松平行性假定（`parallel=FALSE`），允许阈值与协变量相关 $\theta_r = \beta_{r,0} + W\beta_r$

## 3.9 序贯模型
- **不可逆序贯机制**: 从状态 1 依次推进，停比序贯模型 (SRM) $\mathbb{P}(Y=r|Y\ge r, X) = F(\theta_r + X\boldsymbol{\gamma})$ (`sratio()`)
- **续比序贯模型 (CRM)**: $\mathbb{P}(Y>r|Y\ge r, X) = F(\theta_r + X\boldsymbol{\gamma})$ (`cratio()`)，阈值无单调限制
- **案例对比**: 儿童扁桃体不可逆增大中，序贯模型拟合优于累积模型

## 3.10 两步模型
- **建模思路**: 当有序类别可划分为有实际意义的等级集合 $S_1, \dots, S_t$ 时，第一步建模所属集合 $Y \in S_j$，第二步建模集合内具体类别 $Y \le r | Y \in S_j$
- **推广关系**: 当各步潜变量相同时退化为普通累积模型，适用于集合间存在异质性的临床/实验数据
