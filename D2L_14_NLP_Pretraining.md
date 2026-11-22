# 第14章 自然语言处理：预训练 (D2L)

## 14.1 词嵌入 (word2vec)
### 自监督词向量
- **独热编码局限**: 维数等于词表大小 $V$，任意不同词余弦相似度均为 $0$，无法表达语义远近
- **跳元模型 (Skip-Gram)**: 由中心词 $w_c$ 预测窗口内上下文词 $w_o$，$p(o\mid c) = \frac{\exp(\mathbf{u}_o^\top \mathbf{v}_c)}{\sum_i \exp(\mathbf{u}_i^\top \mathbf{v}_c)}$
- **连续词袋 (CBOW)**: 由窗口内上下文平均向量 $\bar{\mathbf{v}} = \frac{1}{m}\sum_{i=1}^m \mathbf{v}_{o_i}$ 预测中心词（忽略词序）

## 14.2 近似训练
### 降低 Softmax 计算代价
- **负采样 (NS)**: 将 $O(Vd)$ 多分类转化为二分类，$\ell_{\text{NS}} = -\log\sigma(s(o,c)) - \sum_{k=1}^K \log\sigma(-s(n_k,c))$，复杂度降至 $O((K+1)d)$
- **层序 Softmax**: 将词置于二叉树叶节点，$p(o\mid c) = \prod_{n\in\text{path}(o)}\sigma(\zeta_n \mathbf{u}_n^\top \mathbf{v}_c)$，路径长 $O(\log V)$（Huffman 树优化高频词）

## 14.3 & 14.4 word2vec 数据集与预训练
### 采样策略与训练实现
- **高频下采样**: 按 $P_{\text{discard}}(w) = \max\left(1 - \sqrt{\frac{t}{f(w)}}, 0\right)$ 丢弃高频词（$t=10^{-4}$），平衡词频分布
- **动态窗口与噪声分布**: 均匀抽取窗口半径，负例按 $f(w)^{0.75}$ 权重抽取并排除真实上下文词
- **掩码损失**: 用两张 `nn.Embedding(V, d)` 表、`torch.bmm` 与带掩码的 `binary_cross_entropy_with_logits` 忽略填充位

## 14.5 全局向量的词嵌入 (GloVe)
### 全局共现矩阵分解
- **加权平方损失**: $L_{\text{GloVe}} = \sum_{i,j:x_{ij}>0} h(x_{ij})(\mathbf{u}_j^\top\mathbf{v}_i + b_i + c_j - \log x_{ij})^2$
- **权重函数**: $h(x) = \min(1, (x/x_{\max})^\alpha)$（常用 $x_{\max}=100, \alpha=0.75$），抑制低频噪声并对高频封顶
- **概率比解释**: 向量内积之差 $(\mathbf{u}_j-\mathbf{u}_k)^\top \mathbf{v}_i$ 直接建模共现概率对数比 $\log(p_{ij}/p_{ik})$

## 14.6 子词嵌入
### fastText 与 BPE
- **fastText**: 将词拆为 $3\sim 6$ 字符 $n$-gram 及带边界整词，词向量为子词之和 $\mathbf{v}_w = \sum_{g\in\mathcal{G}_w}\mathbf{z}_g$，支持未登录词
- **字节对编码 (BPE)**: 从字符出发，每步迭代合并语料频率最高的相邻符号对，控制词表大小并学习切分粒度

## 14.7 词的相似性和类比任务
### 向量空间检索
- **余弦最近邻**: 归一化预训练向量后，用 $\cos(\mathbf{a},\mathbf{b}) = \frac{\mathbf{a}^\top \mathbf{b}}{\|\mathbf{a}\|_2\|\mathbf{b}\|_2}$ 配合 `torch.topk` 检索语义近邻
- **类比偏移**: 对 $a:b::c:d$，计算偏移向量 $\mathbf{q} = \mathbf{v}_b - \mathbf{v}_a + \mathbf{v}_c$ 并检索最相似词

## 14.8 BERT 架构
### 双向编码与输入表示
- **上下文化表示**: 堆叠 Transformer 编码器通过双向自注意力融合左右上下文，解决一词多义
- **输入嵌入**: 单句 `<cls> A <sep>` 或句对 `<cls> A <sep> B <sep>`，相加 $\mathbf{x}_i = \mathbf{e}_{\text{token}}(t_i) + \mathbf{e}_{\text{segment}}(s_i) + \mathbf{e}_{\text{pos}}(i)$
### 双预训练任务
- **掩蔽语言模型 (MLM)**: 随机选 $15\%$ 位置预测原词（$80\%$ 换 `<mask>`、$10\%$ 随机词、$10\%$ 保持原词）
- **下一句预测 (NSP)**: 取 `<cls>` 编码经线性层 + `Tanh` 二分类，判断句对是否为真实连续上下句

## 14.9 & 14.10 BERT 数据集与预训练
### 样本构造与联合训练
- **数据字段**: 构造 `token_ids`、`segments`、`valid_lens`、`pred_positions`、`mlm_weights`、`mlm_labels`、`nsp_labels`
- **双任务损失**: $L = \frac{\sum_{b,i}m_{bi}\ell^{\text{MLM}}_{bi}}{\sum_{b,i}m_{bi}+\varepsilon} + \frac{1}{B}\sum_b \ell^{\text{NSP}}_b$，用 Adam 更新并可提取 `<cls>` 或单词元表示
