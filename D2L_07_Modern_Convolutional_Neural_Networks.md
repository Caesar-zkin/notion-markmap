# 第7章 现代卷积神经网络 (D2L)

## 7.1 深度卷积神经网络（AlexNet）
### 逐层表征与架构
- **层级表征**: 浅层提取边缘颜色纹理，深层组合为局部与整体语义表征
- **结构配置**: 5 个卷积层（$11\times11$ 步幅 4 $\to 5\times5 \to 3\times3\times3$）配 3 个 $3\times3$ 最大汇聚，展平 $6400$ 维接 $4096\to4096\to10$
### 关键改进
- **ReLU 激活**: $\operatorname{ReLU}(x)=\max(0,x)$ 缓解 sigmoid 正饱和区梯度消失
- **正则化与增广**: 全连接层使用 `nn.Dropout(0.5)`，配合翻转、裁切、色彩变化等图像增广

## 7.2 使用块的网络（VGG）
### VGG 块与感受野
- **模块设计**: 若干 `Conv2d(3×3, p=1) → ReLU` 串联后接 $2\times2$（$s=2$）最大汇聚
- **参数效率**: 两个 $3\times3$ 卷积串联等价 $5\times5$ 感受野，权重数由 $25C^2$ 降至 $18C^2$ 且多一次非线性
### VGG-11 配置
- **五块通道配置**: `(1,64), (1,128), (2,256), (2,512), (2,512)`（8 卷积 + 3 全连接），空间尺寸 $224\to7$

## 7.3 网络中的网络（NiN）
### 逐位置小型 MLP
- **NiN 块**: 空间卷积后接两个 $1\times1$ 卷积，$\mathbf y_{i,j}=\operatorname{ReLU}(B\operatorname{ReLU}(A\mathbf z_{i,j}+\mathbf a)+\mathbf b)$
### 全局平均汇聚分类头
- **无全连接分类**: 末块输出通道数等于类别数 $C$，全局平均汇聚 $s_c=\frac{1}{HW}\sum_{i,j}Z_{c,i,j}$ 直接得 logits

## 7.4 含并行连结的网络（GoogLeNet）
### Inception 块的四条路径
- **多尺度并行**: 路径 1 为 $1\times1$ 卷积；路径 2 为 $1\times1\to3\times3$；路径 3 为 $1\times1\to5\times5$；路径 4 为 $3\times3$ 最大汇聚 $\to 1\times1$
- **通道拼接**: 四路输出同高宽，沿通道轴 `torch.cat(outputs, dim=1)`
### 通道瓶颈与分段
- **计算降维**: 先用 $1\times1$ 卷积压缩通道至 $r$，权重数由 $k^2 C_{\mathrm{in}} C_{\mathrm{out}}$ 降至 $C_{\mathrm{in}}r+k^2rC_{\mathrm{out}}$
- **整体结构**: 两段卷积汇聚后接 $2,5,2$ 个 Inception 块，末接全局平均汇聚与线性层

## 7.5 批量规范化
### 小批量统计与仿射变换
- **批统计量**: $\mu_{\mathcal B}=\frac{1}{m}\sum_{i=1}^m x_i$，$v_{\mathcal B}=\frac{1}{m}\sum_{i=1}^m(x_i-\mu_{\mathcal B})^2$
- **归一化与仿射**: $\hat x_i=\frac{x_i-\mu_{\mathcal B}}{\sqrt{v_{\mathcal B}+\epsilon}}$，$y_i=\gamma\hat x_i+\beta$（$\gamma,\beta$ 可学习）
### 统计轴与模式
- **统计轴区分**: 全连接 `BatchNorm1d(D)` 沿批量轴统计；卷积 `BatchNorm2d(C)` 沿批量、高、宽三轴统计
- **训练与评估**: 训练用当前批统计并更新移动平均 $\bar\mu\leftarrow\rho\bar\mu+(1-\rho)\mu_{\mathcal B}$，评估用累计统计

## 7.6 残差网络（ResNet）
### 嵌套函数类与残差映射
- **残差学习**: 令主路径拟合 $g(\mathbf x)=f(\mathbf x)-\mathbf x$，相加点输出 $\mathbf x+g(\mathbf x)$，恒等映射只需 $g=0$
- **梯度通路**: 相加点雅可比矩阵为 $I+J_g(\mathbf x)$，恒等捷径提供无衰减梯度通路
### 基本残差块与 ResNet-18
- **块结构**: `Conv3×3 → BN → ReLU → Conv3×3 → BN` 与捷径相加后接 ReLU；降采样或改通道时捷径用 $1\times1$ 卷积
- **ResNet-18 阶段**: 初始 $7\times7$ 卷积汇聚后，四个阶段各含 2 个残差块（通道 $64,128,256,512$），接全局平均汇聚与分类层

## 7.7 稠密连接网络（DenseNet）
### 稠密块与增长率
- **通道复用**: $\mathbf x_\ell=g_\ell([\mathbf x_0,\mathbf x_1,\ldots,\mathbf x_{\ell-1}])$，输出为全部特征图拼接 $\mathbf y=[\mathbf x_0,\ldots,\mathbf x_L]$
- **增长率 $k$**: 每层新增 $k$ 通道，第 $\ell$ 层输入 $C_0+(\ell-1)k$ 通道，整块输出 $C_0+Lk$ 通道
### 过渡层
- **压缩与降采样**: `BN → ReLU → Conv1×1 → AvgPool2d(2, s=2)` 将通道数减半并缩小空间尺寸
