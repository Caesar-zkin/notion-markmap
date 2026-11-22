# 第4章 多层感知机 (D2L)

## 4.1 多层感知机
### 隐藏层与仿射变换
- **前向计算**: $\mathbf H=\phi(\mathbf X\mathbf W_1+\mathbf b_1)$，$\mathbf O=\mathbf H\mathbf W_2+\mathbf b_2$
- **非线性的必要性**: 若 $\phi$ 为恒等映射，多层退化为单次仿射变换 $\mathbf O=\mathbf X(\mathbf W_1\mathbf W_2)+(\mathbf b_1\mathbf W_2+\mathbf b_2)$
- **多层递推**: $\mathbf H^{(\ell)}=\phi_\ell(\mathbf H^{(\ell-1)}\mathbf W_\ell+\mathbf b_\ell)$
### 激活函数
- **ReLU**: $\operatorname{ReLU}(x)=\max(0,x)$，正半轴导数为 $1$、负半轴为 $0$，`nn.ReLU` 在 $0$ 处取 $0$
- **sigmoid**: $\sigma(x)=(1+e^{-x})^{-1}\in(0,1)$，$\sigma'(x)=\sigma(x)(1-\sigma(x))$，两端饱和
- **tanh**: $\tanh(x)=(e^x-e^{-x})/(e^x+e^{-x})\in(-1,1)$，$\tanh'(x)=1-\tanh^2(x)$，两端饱和
### 通用近似
- **存在性定理**: 紧集 $K\subset\mathbb R^d$ 上，足够宽的单隐藏层网络可一致逼近任意连续函数 $f\in C(K)$
- **异或示例**: $\operatorname{ReLU}(x_1-x_2)+\operatorname{ReLU}(x_2-x_1)=|x_1-x_2|$

## 4.2 多层感知机的从零开始实现
### Fashion-MNIST 参数
- **网络结构**: 输入 $28\times28=784$ 维，隐藏层 $256$ 单元，输出 $10$ 类，共 $203\,530$ 个参数
- **初始化**: 权重用标准差 $0.01$ 的零均值正态分布，偏置初始化为 $0$
### 小批量更新
- **前向与损失**: `reshape` 展平后计算 ReLU 隐藏层与 logits，传给 `nn.CrossEntropyLoss()`
- **优化步骤**: `zero_grad()` $\to$ `loss.backward()` $\to$ `optimizer.step()`，标签类型为 `torch.long`

## 4.3 多层感知机的简洁实现
### 模块化构建
- **Sequential 管线**: `nn.Flatten()` $\to$ `nn.Linear(784, 256)` $\to$ `nn.ReLU()` $\to$ `nn.Linear(256, 10)`
- **线性层存储**: `nn.Linear(d, h)` 权重形状为 $(h,d)$，计算 $\mathbf X\mathbf W^\top+\mathbf b$
- **批量初始化**: 通过 `model.apply(init_weights)` 对所有 `nn.Linear` 初始化权重与偏置

## 4.4 模型选择、欠拟合和过拟合
### 训练误差与泛化误差
- **训练误差**: $\widehat R_{\mathrm{train}}(f)=\frac{1}{n}\sum_{i=1}^n\ell(f(\mathbf x_i),y_i)$
- **泛化误差**: $R_P(f)=\mathbb E_{(\mathbf X,Y)\sim P}[\ell(f(\mathbf X),Y)]$，需用独立测试集估计
### 验证集与 K 折交叉验证
- **数据划分**: 训练集更新参数，验证集选模型与超参数，测试集评估最终方案
- **K 折交叉验证**: $\operatorname{CV}=\frac{1}{K}\sum_{k=1}^K \frac{1}{|\mathcal V_k|}\sum_{i\in\mathcal V_k}\ell(f_{-k}(\mathbf x_i),y_i)$，预处理统计量随每折重拟合
### 多项式回归诊断
- **欠拟合**: 训练误差与验证误差均偏高（模型容量不足）
- **过拟合**: 训练误差持续下降而验证误差升高（过度拟合噪声）

## 4.5 权重衰减
### L2 正则化
- **惩罚目标**: $J(\mathbf w,b)=L(\mathbf w,b)+\frac{\lambda}{2}\|\mathbf w\|_2^2$，通常不惩罚偏置 $b$
- **SGD 更新式**: $\mathbf w\leftarrow(1-\eta\lambda)\mathbf w-\eta\nabla_{\mathbf w}L_{\mathcal B}$，以 $(1-\eta\lambda)$ 收缩已有权重
### 高维线性回归
- **PyTorch 配置**: 通过参数组为 `weight` 设置 `weight_decay`，`bias` 组保持 `weight_decay=0`

## 4.6 暂退法（Dropout）
### 乘性噪声与无偏期望
- **反向暂退法**: $\tilde h=\frac{mh}{1-p}$（$m\sim\operatorname{Bernoulli}(1-p)$），$\mathbb E[\tilde h\mid h]=h$，$\operatorname{Var}(\tilde h\mid h)=\frac{p}{1-p}h^2$
- **正则化机制**: 迫使后续层适应部分隐藏单元随机缺失，降低对固定单元组合的依赖
### 模式切换
- **训练与评估**: `nn.Dropout(p)` 在 `model.train()` 启用掩码，在 `model.eval()` 原样输出

## 4.7 前向传播、反向传播和计算图
### 前向传播与计算图
- **计算依赖**: 节点保存算子或中间结果，预测损失链与权重惩罚链汇入目标 $J$
### 链式法则与梯度累积
- **反向递推**: $\boldsymbol\delta_z=(\mathbf W_2^\top\boldsymbol\delta_o)\odot\phi'(\mathbf z)$，$\nabla_{\mathbf W_2}J=\boldsymbol\delta_o\mathbf h^\top+\lambda\mathbf W_2$
- **多路径相加**: 同一变量经多条路径影响目标时，各路径梯度相加
### 中间值与显存
- **显存开销**: 反向传播需保留前向激活（如 $\mathbf h$）直到反向结束，叶子梯度默认累积需清零

## 4.8 数值稳定性和模型初始化
### 梯度消失与爆炸
- **雅可比连乘**: $\|\partial\mathbf h^{(L)}/\partial\mathbf h^{(\ell)}\|\le\prod_{j=\ell+1}^L\|\mathbf J_j\|$，连乘过小导致梯度消失、过大导致爆炸
- **激活函数影响**: sigmoid 导数最大 $1/4$ 易加剧消失，ReLU 正区间导数为 $1$
### 对称性打破与 Xavier 初始化
- **打破对称**: 独立随机初始化权重防止同层神经元学到相同特征
- **Xavier 折中方差**: $\sigma^2=\frac{2}{n_{\mathrm{in}}+n_{\mathrm{out}}}$，均匀分布 $w_{ij}\sim\mathcal U\bigl[-\sqrt{\frac{6}{n_{\mathrm{in}}+n_{\mathrm{out}}}},\,\sqrt{\frac{6}{n_{\mathrm{in}}+n_{\mathrm{out}}}}\bigr]$

## 4.9 环境和分布偏移
### 三类分布偏移
- **协变量偏移**: $p(\mathbf x)\ne q(\mathbf x)$，但 $p(y\mid\mathbf x)=q(y\mid\mathbf x)$
- **标签偏移**: $p(y)\ne q(y)$，但 $p(\mathbf x\mid y)=q(\mathbf x\mid y)$
- **概念偏移**: $p(y\mid\mathbf x)\ne q(y\mid\mathbf x)$，需用新标注数据更新模型
### 重要性加权
- **协变量纠偏**: $R_p(f)=\mathbb E_q\bigl[\frac{p(\mathbf X)}{q(\mathbf X)}\ell(f(\mathbf X),Y)\bigr]$，权重可由域分类器 $\frac{s(\mathbf x)}{1-s(\mathbf x)}$ 估计
- **标签偏移求解**: 利用混淆矩阵 $\boldsymbol\mu=\mathbf C\mathbf p_Y$ 估计目标标签比例
### 学习方式与环境反馈
- **离线到交互**: 批量学习、在线学习、多臂老虎机、控制与强化学习

## 4.10 实战 Kaggle 比赛：预测房价
### 特征预处理
- **数值特征**: 标准化 $(x-\mu)/s$ 后将缺失值填为 $0$（等价均值填补）
- **类别特征**: 独热编码，缺失值单独成列
### 指标与交叉验证
- **对数 RMSE**: $\operatorname{logRMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^n[\log y_i-\log\max(\hat y_i,1)]^2}$
- **调参与提交**: 按 $K$ 折验证分数选超参数，全量重训后输出 `Id` 与 `SalePrice` CSV
