# 第11章 优化算法 (D2L)

## 11.1 优化和深度学习
### 风险与挑战
- **经验风险 vs 总体风险**: 经验风险 $\widehat{R}(\mathbf{w}) = \frac{1}{n}\sum_i \ell(z_i; \mathbf{w})$，泛化关心总体风险 $R(\mathbf{w}) = \mathbb{E}_z[\ell(z; \mathbf{w})]$
- **局部极小与鞍点**: 驻点处梯度为零，可能是局部极小或鞍点（如 $x^2-y^2$ 在 $(0,0)$ 曲率符号相反）
- **梯度消失**: 激活函数饱和区（如 $\tanh$ 导数 $1-\tanh^2 x \to 0$）导致步进极慢；小批量噪声助于逃离驻点

## 11.2 凸性
### 凸集与凸函数
- **凸集**: 任意 $x, y \in C, \lambda \in [0,1]$ 有 $\lambda x + (1-\lambda)y \in C$；凸集交集仍为凸集
- **凸函数与 Jensen 不等式**: $f(\lambda x + (1-\lambda)y) \le \lambda f(x) + (1-\lambda)f(y)$；期望形式 $f(\mathbb{E}X) \le \mathbb{E}[f(X)]$
### 判据与约束投影
- **一阶/二阶判据**: $f(y) \ge f(x) + \nabla f(x)^\top(y-x)$；二次可微等价于 Hessian 半正定 $\nabla^2 f(x) \succeq 0$
- **全局最优性**: 凸域上凸函数的局部极小即全局极小；严格凸函数最小点唯一
- **投影梯度下降**: $x_{t+1} = \Pi_C(x_t - \eta \nabla f(x_t))$；半径 $r$ 欧氏球投影为 $x\min(1, r/\|x\|_2)$

## 11.3 梯度下降
### 一阶方法与收敛条件
- **梯度下降**: $x_{t+1} = x_t - \eta \nabla f(x_t)$；对 $L$-Lipschitz 梯度函数，需 $0 < \eta < 2/L$ 保证单步下降
- **曲率限制**: 各向异性二次函数（如 $x_1^2 + 2x_2^2$）中，最大曲率方向限制全局步长
### 二阶与辅助策略
- **牛顿法**: $H(x)\Delta = -\nabla f(x)$；Hessian 正定时为二次模型极小方向，但大模型求逆代价高
- **预处理与线搜索**: 对角预处理近似 Hessian 逆；线搜索沿下降方向寻优步长但评估成本高

## 11.4 随机梯度下降 (SGD)
### 无偏估计与收敛界
- **单步更新**: 均匀采样单个样本 $I_t$，$g_t = \nabla f_{I_t}(x_t)$ 满足 $\mathbb{E}[g_t \mid x_t] = \nabla R(x_t)$
- **凸目标误差界**: 取 $\eta_t = \frac{r}{G\sqrt{T}}$，加权平均点 $\bar{x}_T$ 满足 $\mathbb{E}[R(\bar{x}_T)] - R(x^\star) \le \frac{rG}{\sqrt{T}}$
- **采样方式**: 有放回抽 $n$ 次单样本覆盖率约 $1 - (1-1/n)^n \approx 0.63$；无放回打乱每轮遍历全部样本

## 11.5 小批量随机梯度下降
### 方差与硬件吞吐
- **批量梯度**: $g_t = \frac{1}{|\mathcal{B}_t|}\sum_{i\in\mathcal{B}_t}\nabla f_i(x_t)$，协方差由 $\Sigma$ 降至 $\Sigma/b$
- **计算权衡**: 增大批量提升向量化吞吐，但固定遍历预算下减少参数更新次数
- **PyTorch 接口**: `DataLoader(..., batch_size=b, shuffle=True)` 配合 `torch.optim.SGD`

## 11.6 动量法 (Momentum)
### 速度累积与特征分析
- **迭代公式**: $v_t = \beta v_{t-1} + g_t$，$x_t = x_{t-1} - \eta v_t$（$0 \le \beta < 1$），等价指数加权梯度和
- **几何效果**: 一致方向梯度累积加速，高频振荡方向相互抵消，缓解狭长等高线限制
- **二次目标递推**: 特征方向误差满足 $z_t = (1+\beta-\eta\lambda)z_{t-1} - \beta z_{t-2}$；PyTorch 用 `momentum=beta`

## 11.7 AdaGrad 算法
### 逐坐标自适应步长
- **更新规则**: $s_t = s_{t-1} + g_t^2$，$x_t = x_{t-1} - \frac{\eta}{\sqrt{s_t + \epsilon}} \odot g_t$
- **特点与局限**: 高频大梯度坐标步长缩小、稀疏特征步长较大；但 $s_t$ 单调递增导致后期步长过早衰减
- **PyTorch 接口**: `torch.optim.Adagrad`（注意 `eps` 加在平方根外）

## 11.8 RMSProp 算法
### 指数衰减平方梯度
- **更新规则**: $s_t = \gamma s_{t-1} + (1-\gamma)g_t^2$，$x_t = x_{t-1} - \frac{\eta}{\sqrt{s_t+\epsilon}} \odot g_t$（$0 \le \gamma < 1$）
- **半衰期机制**: 旧梯度平方按 $\gamma^k$ 衰减（半衰期 $\ln(1/2)/\ln\gamma$ 步），步长随近期梯度变小而恢复
- **PyTorch 接口**: `torch.optim.RMSprop(..., alpha=gamma)`

## 11.9 Adadelta
### 无全局学习率设计
- **双状态跟踪**: $s_t = \rho s_{t-1} + (1-\rho)g_t^2$，$\Delta_t = \frac{\sqrt{u_{t-1}+\epsilon}}{\sqrt{s_t+\epsilon}} \odot g_t$，$u_t = \rho u_{t-1} + (1-\rho)\Delta_t^2$
- **尺度匹配**: 用参数更新量近期均方根替换全局学习率 $\eta$；PyTorch `Adadelta` 设 `lr=1.0` 对应原式

## 11.10 Adam 与 Yogi
### Adam 算法
- **一阶与二阶矩**: $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$，$s_t = \beta_2 s_{t-1} + (1-\beta_2)g_t^2$（常用 $\beta_1=0.9, \beta_2=0.999$）
- **偏差修正与更新**: $\widehat{m}_t = \frac{m_t}{1-\beta_1^t}$，$\widehat{s}_t = \frac{s_t}{1-\beta_2^t}$，$x_t = x_{t-1} - \eta \frac{\widehat{m}_t}{\sqrt{\widehat{s}_t}+\epsilon}$
### Yogi 改进
- **二阶矩修正**: $s_t = s_{t-1} + (1-\beta_2)g_t^2 \odot \operatorname{sign}(g_t^2 - s_{t-1})$，防止剧烈波动下二阶矩过快衰减

## 11.11 学习率调度器
### 常见衰减策略
- **平方根衰减**: $\eta_t = \eta_0(t+1)^{-1/2}$
- **因子与多步衰减**: $\eta_t = \max(\eta_{\min}, c\eta_{t-1})$；或到达里程碑 epoch 时乘以 $\gamma$（`MultiStepLR`）
- **余弦退火与预热**: $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0-\eta_{\min})(1+\cos(\pi t/T))$；前 $W$ 步线性预热提升早期稳定性
