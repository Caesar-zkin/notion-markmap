---
title: 第3章 线性神经网络 (D2L)
markmap:
  colorFreezeLevel: 2
  maxWidth: 360
  initialExpandLevel: 3
---

# 第3章 线性神经网络

## 3.1 线性回归
### 线性模型
- **标量与向量表达**: $\hat y = \mathbf w^\top\mathbf x + b$
- **矩阵批量表达**: $\hat{\mathbf y} = \mathbf X\mathbf w + b\mathbf 1_n$ ($\mathbf X\in\mathbb R^{n\times d}$)
- **物理意义**: 权重 $w_j$ 表示特征边际贡献，偏置 $b$ 表示零输入基准输出
### 平方损失
- **单样本半平方误差**: $l^{(i)}(\mathbf w,b) = \frac{1}{2}(\hat y^{(i)} - y^{(i)})^2$
- **经验风险最小化**: $L(\mathbf w,b) = \frac{1}{n}\sum_{i=1}^n l^{(i)}(\mathbf w,b)$
### 最小二乘解析解
- **增广矩阵目标**: $f(\mathbf w) = \frac{1}{2}\lVert \mathbf X\mathbf w - \mathbf y\rVert_2^2$
- **正规方程**: $\mathbf X^\top\mathbf X\mathbf w = \mathbf X^\top\mathbf y$
- **闭式解** ($\mathbf X$ 列满秩): $\mathbf w^* = (\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top\mathbf y$
### 小批量随机梯度下降 (Mini-batch SGD)
- **残差定义**: $e_i = \mathbf w^\top\mathbf x^{(i)} + b - y^{(i)}$
- **权重更新**: $\mathbf w \leftarrow \mathbf w - \frac{\eta}{|\mathcal B|}\sum_{i\in\mathcal B} e_i\mathbf x^{(i)}$
- **偏置更新**: $b \leftarrow b - \frac{\eta}{|\mathcal B|}\sum_{i\in\mathcal B} e_i$
- **超参数 vs 模型参数**: 学习率 $\eta$、批量大小 $|\mathcal B|$ vs 权重 $\mathbf w$、偏置 $b$
### 矢量化与概率解释
- **矢量化加速**: 矩阵乘法 $\mathbf X\mathbf w + b$ 替代 Python 循环，广播机制加偏置
- **正态噪声与极大似然**: 设 $\epsilon\sim\mathcal N(0,\sigma^2)$，最小化负对数似然 $-\log p(\mathbf y\mid\mathbf X)$ 等价于最小二乘估计
- **神经网络视角**: 输入维数 $d$ 到单一输出的全连接单层神经网络

## 3.2 线性回归的从零实现
### 合成数据生成
- **真实参数**: $\mathbf w_0 = (2, -3.4)^\top,\ b_0 = 4.2$
- **数据分布**: $\mathbf X\in\mathbb R^{1000\times 2}$ 标准正态采样，加 $\mathcal N(0, 0.01^2)$ 噪声
### 小批量读取与张量形状
- **打乱与切片**: 每轮随机打乱索引，批大小 $|\mathcal B| = 10$
- **形状对齐**: $\mathbf X_{\mathcal B}:(m,2),\ \mathbf w:(2,1),\ \hat{\mathbf y}/ \mathbf y:(m,1)$（防止 $(m,1)$ 与 $(m,)$ 广播成 $(m,m)$）
### 初始化与训练循环
- **参数初始化**: $\mathbf w\sim\mathcal N(0, 0.01^2)$，$b = 0$
- **反向传播与步进**: 批损失求和反传后除以 $m$（等价于平均损失反传），$\eta = 0.03$ 训练 3 轮

## 3.3 线性回归的简洁实现 (PyTorch)
### 数据与模型 API
- **数据管道**: `TensorDataset(features, labels)` + `DataLoader(batch_size=10, shuffle=True)`
- **模型构建**: `nn.Sequential(nn.Linear(2, 1))`（权重形状为 `(1, 2)`）
- **参数初始化**: `nn.init.normal_(weight, 0, 0.01)`、`nn.init.zeros_(bias)`
### 损失函数与优化器
- **均方误差**: `nn.MSELoss()`（默认求均值，不含 $\frac{1}{2}$ 系数）
- **优化器更新**: `torch.optim.SGD(model.parameters(), lr=0.03)` (`zero_grad` $\rightarrow$ `backward` $\rightarrow$ `step`)

## 3.4 Softmax 回归
### 分类与全连接 Logits
- **独热编码 (One-hot)**: $q$ 类标签表示为长度 $q$ 的单位向量（如猫 `(1,0,0)`、鸡 `(0,1,0)`、狗 `(0,0,1)`）
- **仿射变换**: $\mathbf O = \mathbf X\mathbf W + \mathbf b$，其中 $\mathbf X:(m,d),\ \mathbf W:(d,q),\ \mathbf b:(1,q),\ \mathbf O:(m,q)$
### Softmax 算子
- **行归一化概率**: $\hat y_j = \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)}$，满足 $\sum_{j=1}^q \hat y_j = 1$
- **单调保序性**: $\operatorname*{argmax}_j \hat y_j = \operatorname*{argmax}_j o_j$
### 交叉熵损失与梯度
- **单样本交叉熵**: $l(\mathbf y, \hat{\mathbf y}) = -\sum_{j=1}^q y_j\log\hat y_j = -\log\hat y_c$（等价于负对数似然）
- **关于 Logits 的梯度**: $\frac{\partial l}{\partial o_j} = \hat y_j - y_j$（预测概率减去真实标签）
### 信息论基础与评估
- **自信息与香农熵**: $I(j) = -\log P(j)$，熵 $H(P) = -\sum_j P(j)\log P(j)$
- **交叉熵下界**: $H(P,Q) = -\sum_j P(j)\log Q(j) \ge H(P)$（当 $Q=P$ 时取等）
- **分类准确率**: $\mathrm{accuracy} = \frac{1}{n}\sum_{i=1}^n \mathbf 1\{\hat c^{(i)} = c^{(i)}\}$

## 3.5 图像分类数据集 (Fashion-MNIST)
### 数据集结构
- **规模与尺寸**: 训练集 $60,000$ 张，测试集 $10,000$ 张，单图形状 $(1, 28, 28)$
- **10 类标签**: `0:T恤` `1:裤子` `2:套衫` `3:连衣裙` `4:外套` `5:凉鞋` `6:衬衫` `7:运动鞋` `8:包` `9:短靴`
### 读取与变换管道
- **图像预处理**: `transforms.ToTensor()` 转为通道在前浮点张量并归一化至 $[0,1]$
- **批量加载**: `batch_size = 256`，特征批次形状 $(256, 1, 28, 28)$，标签形状 $(256,)$

## 3.6 Softmax 回归的从零实现
### 模型与按行归一化
- **图像展平**: $(1, 28, 28) \rightarrow 784$ 维向量，参数 $\mathbf W\in\mathbb R^{784\times 10},\ \mathbf b\in\mathbb R^{1\times 10}$
- **广播归一化**: 沿类别维度求和并保持维度 `keepdim=True`：$\hat Y_{ij} = \frac{\exp(O_{ij})}{\sum_k \exp(O_{ik})}$
### 高级索引与训练评估
- **真实类概率提取**: 用花式索引 `y_hat[range(len(y_hat)), y]` 取出对应类别的预测概率计算 $-\log \hat y_c$
- **实验配置**: 批大小 $256$，学习率 $0.1$，训练 $10$ 轮，统计训练损失、训练准确率与测试集准确率

## 3.7 Softmax 回归的简洁实现 (PyTorch)
### 网络层与数值稳定性
- **模型结构**: `nn.Sequential(nn.Flatten(), nn.Linear(784, 10))`
- **LogSumExp 技巧**: 令 $a = \max_k o_k$，平移指数防止上溢/下溢：$\log\hat y_j = (o_j - a) - \log\sum_k \exp(o_k - a)$
### 交叉熵 API 与训练闭环
- **内置稳定损失**: `nn.CrossEntropyLoss(reduction="none")` 直接传入未归一化 `logits` 与类别索引 `targets`
- **训练与推断分离**: `model.train()` 反传更新；`model.eval()` + `torch.no_grad()` 下用 `logits.argmax(dim=1)` 计算测试准确率
