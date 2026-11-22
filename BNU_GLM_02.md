# 第2章 一变量广义线性模型

## 2.1 二分类模型与 2.2 性质
### 三种常用模型
- **Logit 模型**: $\mathbb{E}(Y|\mathbf{X}) = \frac{\exp(\mathbf{Z}(\mathbf{X})\boldsymbol{\beta}^\top)}{1 + \exp(\mathbf{Z}(\mathbf{X})\boldsymbol{\beta}^\top)}$（对称、优势比简洁）
- **Probit 模型**: $\mathbb{E}(Y|\mathbf{X}) = \Phi(\mathbf{Z}(\mathbf{X})\boldsymbol{\beta}^\top)$（基于正态潜变量）
- **Cloglog 模型**: $\mathbb{E}(Y|\mathbf{X}) = 1 - \exp(-\exp(\mathbf{Z}(\mathbf{X})\boldsymbol{\beta}^\top))$（非对称，适合稀有事件）
### 分布与极大似然估计
- **条件分布**: $Y \sim B(1, h(\mathbf{Z}(\mathbf{X})\boldsymbol{\beta}^\top))$，条件方差 $h(\eta)(1 - h(\eta))$
- **对数似然**: $l(\boldsymbol{\beta}) = \sum_{i=1}^n [Y_i \log h(\eta_i) + (1 - Y_i)\log(1 - h(\eta_i))]$

## 2.3 优势 (Odds) 与 2.4 迭代算法
- **优势与优势比**: $O(\mathbf{x}) = \frac{\mathbb{P}(Y=1|\mathbf{x})}{\mathbb{P}(Y=0|\mathbf{x})}$；Logit 模型中 $O(\mathbf{x}) = \exp(\mathbf{Z}(\mathbf{x})\boldsymbol{\beta}^\top)$，优势比 $\exp((z^{(1)}-z^{(2)})\boldsymbol{\beta}^\top)$
- **得分函数**: $\mathbf{s}(\boldsymbol{\beta}) = \sum_{i=1}^n \frac{Y_i - h(\eta_i)}{h(\eta_i)(1 - h(\eta_i))}h'(\eta_i)\mathbf{Z}_i$
- **Fisher 信息矩阵**: $\mathbf{I}(\boldsymbol{\beta}) = \sum_{i=1}^n \frac{(h'(\eta_i))^2}{h(\eta_i)(1 - h(\eta_i))}\mathbf{Z}_i^\top \mathbf{Z}_i$
- **Fisher 得分迭代**: $\hat{\boldsymbol{\beta}}^{(k+1)} = \hat{\boldsymbol{\beta}}^{(k)} + \mathbf{s}(\hat{\boldsymbol{\beta}}^{(k)})(\mathbf{I}(\hat{\boldsymbol{\beta}}^{(k)}))^{-1}$

## 2.5 假设检验与 2.6 拟合优度
### 参数假设检验 ($H_0: \mathbf{A}\boldsymbol{\beta}^\top = \mathbf{a}$)
- **对数似然比检验**: $\lambda = -2(l(\tilde{\boldsymbol{\beta}}) - l(\hat{\boldsymbol{\beta}})) \stackrel{a}{\sim} \chi^2(r)$
- **Wald 检验**: $W = (\mathbf{A}\hat{\boldsymbol{\beta}}^\top - \mathbf{a})^\top (\mathbf{A}\mathbf{I}(\hat{\boldsymbol{\beta}})^{-1}\mathbf{A}^\top)^{-1}(\mathbf{A}\hat{\boldsymbol{\beta}}^\top - \mathbf{a}) \stackrel{a}{\sim} \chi^2(r)$
- **得分检验**: $U = \mathbf{s}(\tilde{\boldsymbol{\beta}})\mathbf{I}(\tilde{\boldsymbol{\beta}})^{-1}\mathbf{s}(\tilde{\boldsymbol{\beta}})^\top \stackrel{a}{\sim} \chi^2(r)$
- **单分量 $z$ 检验**: $T_i = \frac{\hat{\beta}_i}{\sqrt{a_{ii}}} \stackrel{a}{\sim} N(0, 1)$
### 分组数据拟合优度
- **皮尔逊统计量**: $\chi^2 = \sum_{k=1}^g \frac{n_k(\bar{Y}_k - h(\mathbf{Z}(\mathbf{v}_k)\hat{\boldsymbol{\beta}}^\top))^2}{h(\dots)(1 - h(\dots))} \stackrel{a}{\sim} \chi^2(g-p)$
- **偏差统计量 (Deviance)**: $D = -2\sum_{i=1}^g n_i(f_i(h(\mathbf{Z}(\mathbf{v}_i)\hat{\boldsymbol{\beta}}^\top)) - f_i(\bar{Y}_i))$，越小越好

## 2.7 剖腹产感染案例与 2.8 婚外情案例
- **变量选择与 AIC**: $\text{AIC} = 2p - 2l(\hat{\boldsymbol{\beta}})$，用 `step()` 双向/向前/向后筛选最优设计向量
- **极端数据与经验样本**: 全 0/全 1 导致标准误爆炸与不收敛，引入先验经验修正后恢复正常
- **条件边际效应**: 含交互项时不能直接读单系数，需算线性预测子对变量的偏导
- **离散解释变量**: 有序/多水平离散变量（如宗教信仰）应做哑变量编码并用 `anova(..., test="Chisq")` 检验

## 2.9 模拟研究与 ROC 曲线
- **阈值预报与误分率**: $\hat{Y}_c = 1_{\{h(\mathbf{Z}\boldsymbol{\beta}) > c\}}$，样本误分率受正负样本比例影响
- **ROC 曲线 (`pROC::roc`)**: 以 $(FPR, TPR)$ 为坐标，选取最接近 $(0, 1)$ 的点作为最优阈值

## 2.10 非负整数值响应 GLM
- **Poisson 回归**: `family = poisson`，要求条件期望等于条件方差 $\text{var}(Y|\mathbf{X}) = h(\mathbf{Z}\boldsymbol{\beta}^\top)$
- **过离散检验**: Pearson 离散度 $\varphi = \frac{1}{n-p}\sum \frac{(Y_i - \hat{\mu}_i)^2}{\hat{\mu}_i}$ 显著大于 1 时存在过离散（如婚外情 $\hat{\varphi}=7.61$）
- **负二项回归 (`glmmTMB(..., family=nbinom2)`)**: $\text{var}(Y|\mathbf{X}) = \mu(1 + \mu/\theta)$，通过离散参数 $\theta$ 捕捉过离散
- **零膨胀负二项 (ZINB)**: $Y = \xi \eta$ (`ziformula = ~1`)，同时刻画结构零与过离散
