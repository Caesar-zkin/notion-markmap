# 第15章 自然语言处理：应用 (D2L)

## 15.1 情感分析及数据集
### IMDb 二分类任务
- **数据构成**: 训练集与测试集各 $25,000$ 条电影评论，正面标签 $1$、负面标签 $0$，正负各半
- **预处理**: 按词切分并设 `min_freq=5` 建词表，通过 `d2l.truncate_pad` 统一截断/补齐至 `num_steps=500`

## 15.2 情感分析：循环神经网络 (BiLSTM)
### 双向编码与首尾拼接
- **网络结构**: 冻结预训练 $100$ 维 GloVe 嵌入 $\to$ 2 层双向 LSTM（隐藏维 $H=100$）
- **序列表示**: 拼接首尾时间步隐藏状态 $\mathbf{r} = [\mathbf{h}_1; \mathbf{h}_T] \in \mathbb{R}^{4H}$，经线性层输出二分类 logits

## 15.3 情感分析：卷积神经网络 (textCNN)
### 一维卷积与最大时间汇聚
- **通道表示**: 拼接可训练与冻结两份 GloVe 嵌入表，形状转为 `[B, D, T]` 送入 `nn.Conv1d`
- **多核并行**: 并行使用宽度 $3, 4, 5$ 的一维卷积（各 $100$ 通道）捕捉不同长度 $n$-gram 模式
- **汇聚分类**: 经 ReLU 与最大时间汇聚 $r_c = \max_t \operatorname{ReLU}(Y_{c,t})$ 得 `[B, 300]`，过 Dropout 后线性分类

## 15.4 自然语言推断 (NLI) 与数据集
### 句对关系与 SNLI
- **三类关系**: 给定前提 $P$ 判断假设 $H$：蕴涵（0，可推出）、矛盾（1，互斥）、中性（2，无必然关系）
- **SNLI 数据**: 超 $50$ 万带标签英语句对，数据迭代器返回 `((premises, hypotheses), labels)`

## 15.5 自然语言推断：可分解注意力
### Attend - Compare - Aggregate
- **关注 (Attend)**: 共享 MLP $f$ 计算交叉匹配分数 $e_{ij} = f(\mathbf{a}_i)^\top f(\mathbf{b}_j)$，双向 softmax 得软对齐摘要 $\boldsymbol{\beta}_i, \boldsymbol{\alpha}_j$
- **比较 (Compare)**: 共享 MLP $g$ 分别比较原词与对齐摘要：$\mathbf{v}_{A,i} = g([\mathbf{a}_i; \boldsymbol{\beta}_i])$，$\mathbf{v}_{B,j} = g([\mathbf{b}_j; \boldsymbol{\alpha}_j])$
- **聚合 (Aggregate)**: 两段分别求和得到 $\mathbf{v}_A, \mathbf{v}_B$，拼接后经 MLP $h([\mathbf{v}_A; \mathbf{v}_B])$ 输出三分类 logits

## 15.6 针对下游任务微调 BERT
### 序列级与词元级任务
- **序列级任务**: 情感分析、NLI 等取 `<cls>` 位置隐藏表示接任务头，端到端微调编码器
- **词元级标注**: 命名实体识别等对每个有效词元位置 $\mathbf{h}_i$ 共享线性分类头 $W\mathbf{h}_i + \mathbf{b}$
- **抽取式问答**: SQuAD 任务用两个独立线性头预测段落起止分数 $s_i, e_i$，推断时在 $i\le j$ 跨度最大化 $s_i+e_j$

## 15.7 自然语言推断：微调 BERT
### 句对打包与端到端微调
- **句对打包**: 预留 3 个特殊词元位，若 $|P|+|H| > L_{\max}-3$ 则交替截断较长段，拼为 `<cls> P <sep> H <sep>`
- **模型复用**: 加载预训练 `bert.small`（$H=256, 4$ 头, $2$ 层），取 `bert.hidden(encoded[:, 0, :])` 接 3 类线性头并用 Adam 微调
