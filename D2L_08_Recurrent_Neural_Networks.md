# 第8章 循环神经网络 (D2L)

## 8.1 序列模型
### 自回归与固定历史窗口
- **自回归建模**: 用过去观测预测当前 $P(x_t\mid x_{1:t-1})$，截断为长度 $\tau$ 的滑动窗口 $\hat x_t=f(x_{t-\tau},\ldots,x_{t-1})$
### 隐变量与马尔可夫假设
- **隐状态汇总**: $h_t=g(h_{t-1},x_t)$，参数量独立于序列长度
- **$\tau$ 阶马尔可夫**: $P(x_t\mid x_{1:t-1})=P(x_t\mid x_{t-\tau:t-1})$，一阶情形 $P(x_{1:T})=P(x_1)\prod_{t=2}^T P(x_t\mid x_{t-1})$
### 正弦序列与滚动预测
- **一步 vs 多步**: 一步预测输入真实历史；多步滚动预测把预测值放回窗口，误差随步数累积

## 8.2 文本预处理
### 读取与词元化
- **清洗与切分**: 连续非字母替换为空格并转小写；词级切分保留词边界，字符级词表更小可表新词
### 词表与语料索引
- **Vocab 映射**: 按频降序构造，`token_to_idx` 与 `idx_to_token` 双向映射，未登录词映射为 `<unk>`（索引 $0$）

## 8.3 语言模型和数据集
### 联合概率与极大似然
- **链式分解**: $P(x_1,\ldots,x_T)=\prod_{t=1}^T P(x_t\mid x_{1:t-1})$，最大化对数似然等价最小化负对数概率之和
### n 元语法与拉普拉斯平滑
- **平滑估计**: $\widehat P(b\mid a)=\frac{C(a,b)+\alpha}{\sum_{b'}C(a,b')+\alpha V}$，避免未见组合概率为零，但 $n$ 增大时组合数按 $V^n$ 膨胀
### 齐普夫定律
- **经验幂律**: 词频排名 $i$ 与频数满足 $n_i\propto i^{-\alpha}$（$\log n_i=-\alpha\log i+c$），高频词集中、低频长尾众多
### 随机采样与顺序分区
- **随机采样**: 子序列打乱成批，相邻批次同位置无连续性，每批需重置隐状态
- **顺序分区**: 相邻批次同一行连续衔接，可跨批传递隐状态（`X, Y` 形状同为 $(B,T)$）

## 8.4 循环神经网络
### 带隐状态的循环层
- **隐状态与输出**: $\mathbf H_t=\phi(\mathbf X_t\mathbf W_{xh}+\mathbf H_{t-1}\mathbf W_{hh}+\mathbf b_h)$，$\mathbf O_t=\mathbf H_t\mathbf W_{hq}+\mathbf b_q$
- **参数复用**: 同一组权重跨时间步共享，可将 $\mathbf X_t,\mathbf H_{t-1}$ 拼接后用单次矩阵乘法计算
### 字符级语言模型与困惑度
- **独热输入**: `one_hot` 向量输入每步预测下一字符分布
- **困惑度 (PPL)**: $\operatorname{PPL}=\exp\bigl(-\frac{1}{N}\sum_{t=1}^N\log p_\theta(x_t\mid x_{1:t-1})\bigr)$，均匀预测为 $V$，完美预测为 $1$

## 8.5 循环神经网络的从零开始实现
### 参数与前向传播
- **张量布局**: 将 $(B,T)$ 转置独热为 $(T,B,V)$，沿时间维 $\tanh$ 更新 $(B,h)$ 状态并拼接输出 $(TB,V)$
### 前缀预热与字符生成
- **两阶段生成**: 先用前缀真实字符更新隐状态，再循环取 logits 最大索引作为下一输入
### 梯度裁剪与跨批状态
- **全局二范数裁剪**: 若 $\|\mathbf g\|_2>\theta$，则 $\mathbf g\leftarrow\frac{\theta}{\|\mathbf g\|_2}\mathbf g$，防止梯度爆炸
- **状态分离**: 顺序分区跨批传递状态前需调用 `detach()` 截断计算图

## 8.6 循环神经网络的简洁实现
### PyTorch 循环层与线性输出
- **API 形状**: `nn.RNN(V, h)` 输入 $(T,B,V)$ 与初态 $(1,B,h)$，返回 `hidden_seq` $(T,B,h)$ 与 `last_state` $(1,B,h)$
- **读出层**: `nn.Linear(h, V)` 将展平后的 $(TB,h)$ 映射为 $(TB,V)$ logits

## 8.7 通过时间反向传播 (BPTT)
### 展开递归与链式法则
- **总导数展开**: $D_t=\frac{\mathrm dh_t}{\mathrm dw_h}=B_t+A_tD_{t-1}=\sum_{i=1}^t(A_t\cdots A_{i+1})B_i$，长链雅可比连乘易导致梯度消失或爆炸
### 完整传播与截断策略
- **完整 BPTT**: 保留整段依赖，显存高且易遇长链数值问题
- **固定截断**: 反向限制在有限窗口内，前向仍传递状态值
- **随机截断**: 以 $\xi_t\in\{0, 1/\pi_t\}$（$\mathbb E[\xi_t]=1$）无偏估计 $D_t$，但方差可能较高
### 线性循环层梯度
- **时间反向递推**: $\boldsymbol\delta_t=\mathbf W_{qh}^\top\mathbf e_t+\mathbf W_{hh}^\top\boldsymbol\delta_{t+1}$，各步梯度求和 $\nabla_{\mathbf W_{hh}}L=\sum_t\boldsymbol\delta_t\mathbf h_{t-1}^\top$
