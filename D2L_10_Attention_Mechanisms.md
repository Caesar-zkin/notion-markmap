# 第10章 注意力机制 (D2L)

## 10.1 注意力提示
### 自主与非自主提示
- **自主提示**: 来自当前目标的主动选择（如寻找特定书籍），对应查询 $\mathbf{q}$
- **非自主提示**: 来自外界显著刺激（如醒目颜色），对应键 $\mathbf{k}_i$
- **注意力汇聚**: 根据查询 $\mathbf{q}$ 与键 $\mathbf{k}_i$ 的交互动态分配权重给值 $\mathbf{v}_i$；均值汇聚为等权重
- **热图可视化**: 行对应查询、列对应键，如 `torch.eye(10).reshape(1, 1, 10, 10)` 仅对角线匹配

## 10.2 Nadaraya-Watson 核回归
### 非参数与参数化核回归
- **核回归公式**: $\hat{y}(x) = \sum_{i=1}^n \alpha_i(x) y_i$，其中权重 $\alpha_i(x) = \frac{K(x-x_i)}{\sum_j K(x-x_j)}$
- **高斯核形式**: $K(u) = \frac{1}{\sqrt{2\pi}}\exp(-u^2/2)$ 化简为 $\alpha_i(x) = \operatorname{softmax}_i(-\frac{1}{2}(x-x_i)^2)$
- **可学习核回归**: 评分改为 $-\frac{1}{2}[w(x-x_i)]^2$；$|w|$ 越大越集中于近邻，$w=0$ 退化为平均汇聚
- **训练细节**: 键值对排除查询自身以防直接读取标签，用 `torch.bmm` 执行批量加权汇聚

## 10.3 注意力评分函数
### 评分与加性注意力
- **通用注意力**: $\alpha_i(\mathbf{q}) = \operatorname{softmax}_i(a(\mathbf{q}, \mathbf{k}_i))$，$\operatorname{Attention}(\mathbf{q}, K, V) = \sum_i \alpha_i(\mathbf{q})\mathbf{v}_i$
- **加性注意力**: $a(\mathbf{q}, \mathbf{k}) = \mathbf{w}_v^\top \tanh(W_q \mathbf{q} + W_k \mathbf{k})$，支持不同输入维度的查询和键
### 缩放点积与掩蔽
- **缩放点积**: $a(\mathbf{q}, \mathbf{k}) = \frac{\mathbf{q}^\top \mathbf{k}}{\sqrt{d_k}}$；除 $\sqrt{d_k}$ 使方差保持为 $1$，缓解高维 softmax 饱和
- **矩阵形式**: $\operatorname{Attention}(Q, K, V) = \operatorname{softmax}_{\text{row}}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$
- **掩码机制**: 加法掩码 $M$ 在可见位置为 $0$、屏蔽位置为 $-\infty$（如填充掩码、因果掩码）

## 10.4 Bahdanau 注意力
### 动态序列对齐
- **键值来源**: 编码器所有时刻隐藏状态 $\mathbf{h}_1, \ldots, \mathbf{h}_T$ 同时作为键和值
- **动态上下文**: 解码时刻 $t$ 用 $\mathbf{s}_{t-1}$ 查询，$\mathbf{c}_t = \sum_i \alpha_{t,i} \mathbf{h}_i$（加性注意力）
- **解码更新**: 将上一输出词元嵌入与 $\mathbf{c}_t$ 拼接送入 GRU；训练用教师强制，屏蔽源序列填充

## 10.5 多头注意力
### 并行子空间投影
- **多头公式**: $H_i = \operatorname{Attention}(QW_i^Q, KW_i^K, VW_i^V)$，$Y = \operatorname{Concat}(H_1, \ldots, H_h)W^O$
- **张量重排**: 隐藏维度 $d$ 需被头数 $h$ 整除，将 `[B, L, d]` 变形为 `[B, h, L, d/h]` 并行计算
- **PyTorch 接口**: `nn.MultiheadAttention(embed_dim, num_heads, dropout=..., batch_first=True)`

## 10.6 自注意力和位置编码
### 自注意力复杂度对比
- **卷积层**: 每层计算量 $O(knd^2)$，顺序操作 $O(1)$，最长路径 $O(n/k)$
- **循环层**: 每层计算量 $O(nd^2)$，顺序操作 $O(n)$，最长路径 $O(n)$
- **自注意力汇聚**: 每层计算量 $O(n^2d)$，顺序操作 $O(1)$，最长路径 $O(1)$（无位置编码时置换等变）
### 正弦位置编码
- **编码定义**: $P_{i,2j} = \sin(i\omega_j)$，$P_{i,2j+1} = \cos(i\omega_j)$，其中 $\omega_j = 10000^{-2j/d}$
- **相对位置性质**: 固定偏移 $\delta$ 的位置编码可表示为与绝对位置 $i$ 无关的 $2\times 2$ 旋转矩阵线性变换

## 10.7 Transformer
### 编码器与规范化
- **编码器层**: 多头自注意力 + 基于位置的前馈网络 $\operatorname{FFN}(x) = W_2\operatorname{ReLU}(W_1x+b_1)+b_2$
- **残差与层规范化**: 后置规范化 $\operatorname{LayerNorm}(X + \operatorname{Dropout}(\operatorname{Sublayer}(X)))$，在特征维统计
- **嵌入缩放**: 词元嵌入乘以 $\sqrt{d}$ 后再与位置编码相加
### 解码器与训练推理
- **三层结构**: 掩蔽多头自注意力（因果掩码 $j \le i$）$\to$ 编码器-解码器交叉注意力 $\to$ FFN
- **训练与推理**: 训练用右移目标序列并行计算；推理逐步生成并缓存每层历史键值表示
