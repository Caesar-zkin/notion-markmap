# 第9章 现代循环神经网络 (D2L)

## 9.1 门控循环单元（GRU）
### 重置门与更新门
- **门控计算**: $\mathbf R_t=\sigma(\mathbf X_t\mathbf W_{xr}+\mathbf H_{t-1}\mathbf W_{hr}+\mathbf b_r)$，$\mathbf Z_t=\sigma(\mathbf X_t\mathbf W_{xz}+\mathbf H_{t-1}\mathbf W_{hz}+\mathbf b_z)$
- **候选与更新**: $\widetilde{\mathbf H}_t=\tanh(\mathbf X_t\mathbf W_{xh}+(\mathbf R_t\odot\mathbf H_{t-1})\mathbf W_{hh}+\mathbf b_h)$，$\mathbf H_t=\mathbf Z_t\odot\mathbf H_{t-1}+(1-\mathbf Z_t)\odot\widetilde{\mathbf H}_t$
### 信息保留与实现
- **门控作用**: $\mathbf R_t\to 0$ 遗忘旧状态，$\mathbf Z_t\to 1$ 直接保留旧状态；PyTorch 使用 `nn.GRU(V, h)`

## 9.2 长短期记忆网络（LSTM）
### 输入门、遗忘门与输出门
- **三门与候选**: $\mathbf I_t=\sigma(A_i)$、$\mathbf F_t=\sigma(A_f)$、$\mathbf O_t=\sigma(A_o)$、$\widetilde{\mathbf C}_t=\tanh(A_c)$
- **双状态更新**: $\mathbf C_t=\mathbf F_t\odot\mathbf C_{t-1}+\mathbf I_t\odot\widetilde{\mathbf C}_t$，$\mathbf H_t=\mathbf O_t\odot\tanh(\mathbf C_t)$
### 记忆通路与实现
- **长期记忆**: 当 $\mathbf F_t\to 1$ 且 $\mathbf I_t\to 0$ 时，记忆元与梯度可沿直接通路跨步保留
- **PyTorch 接口**: `nn.LSTM(V, h)` 输出 $(T,B,h)$ 与状态元组 `(h_n, c_n)`（均为 $(1,B,h)$）

## 9.3 深度循环神经网络
### 按层堆叠
- **层间与时间递推**: $\mathbf H_t^{(\ell)}=\phi_\ell(\mathbf H_t^{(\ell-1)}\mathbf W_{xh}^{(\ell)}+\mathbf H_{t-1}^{(\ell)}\mathbf W_{hh}^{(\ell)}+\mathbf b_h^{(\ell)})$，$1\le\ell\le L$
- **多层参数**: `nn.LSTM(V, h, num_layers=2)` 返回顶层输出 $(T,B,h)$，末状态 `h_n, c_n` 为 $(2,B,h)$

## 9.4 双向循环神经网络
### 两个方向的递推
- **前向与后向拼接**: $\mathbf H_t=[\overrightarrow{\mathbf H}_t;\overleftarrow{\mathbf H}_t]\in\mathbb R^{B\times 2h}$，`bidirectional=True` 输出 $(T,B,2h)$、末状态 $(2L,B,h)$
### 因果约束
- **适用场景**: 适合序列编码、填空与标注；不适合自回归下一词元预测（推理时未来不可见）

## 9.5 机器翻译与数据集
### 英法双语语料与特殊词元
- **预处理**: 统一小写、标点前补空格并词级切分，源与目标分别建词表
- **特殊词元**: `<pad>` 填充、`<bos>` 解码起始、`<eos>` 序列结束、`<unk>` 未知词
### 小批量与有效长度
- **返回格式**: 源/目标索引 $(B,T)$ 与对应非 `<pad>` 有效长度 $(B,)$，用于屏蔽填充位置

## 9.6 编码器-解码器架构
### 编码与条件生成
- **条件概率**: $P(y_{1:T'}\mid c)=\prod_{t=1}^{T'}P(y_t\mid y_{1:t-1},c)$，编码器将变长源序列映射为上下文表示 $c$
### PyTorch 模块接口
- **标准流程**: `enc_outputs = encoder(enc_X, *args)` $\to$ `dec_state = decoder.init_state(enc_outputs, *args)` $\to$ `logits, dec_state = decoder(dec_X, dec_state)`

## 9.7 序列到序列学习（seq2seq）
### GRU 编码器与解码器
- **上下文拼接**: 解码器取编码器末状态顶层 `state[-1]` 沿时间广播并与目标嵌入拼接
### 遮蔽交叉熵与训练推理
- **遮蔽损失**: `MaskedSoftmaxCELoss` 用 $m_{it}=\mathbf 1[t\le\ell_i]$ 屏蔽 `<pad>` 位置损失
- **强制教学 (Teacher Forcing)**: 训练时输入右移一位的真实目标（如 `[<bos>, va, !]`），推理时逐步自回归直至 `<eos>`
### BLEU 评价指标
- **公式**: $\operatorname{BLEU}=\exp\bigl(\min(0,1-\frac{L_{\mathrm{ref}}}{L_{\mathrm{pred}}})\bigr)\prod_{n=1}^k p_n^{1/2^n}$，惩罚过短预测并统计裁剪 $n$-gram 精确率

## 9.8 束搜索
### 贪心搜索与穷举搜索
- **复杂度对比**: 贪心搜索每步选最大概率词元，代价 $\mathcal O(VT')$；穷举搜索考察 $V^{T'}$ 种组合
### 束宽与长度归一化
- **束搜索**: 每步保留累计对数概率最高的 $k$ 个前缀，扩展代价 $\mathcal O(kVT')$（$k=1$ 退化为贪心）
- **长度归一化**: $s(y_{1:L})=\frac{1}{L^\alpha}\sum_{t=1}^L\log P(y_t\mid y_{1:t-1},c)$（$\alpha\approx0.75$），避免偏向过短序列
