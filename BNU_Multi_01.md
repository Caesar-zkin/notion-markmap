# Chapter 1 矩阵运算

## 01 基础概念
### 线性空间与子空间
- **线性空间 ($\mathcal{H}$)**: $\mathcal{R}_n$ 的子集对向量加法和数乘封闭
- **列空间 ($\mathcal{M}(\boldsymbol{A})$)**: 矩阵 $\boldsymbol{A}$ 的列向量张成的子空间，维数 $\dim(\mathcal{M}(\boldsymbol{A})) = \operatorname{rank}(\boldsymbol{A})$
- **线性相关与基**: 存在不全为零的 $\alpha_i$ 使 $\sum \alpha_i \boldsymbol{a}_i = \boldsymbol{0}$ 则相关；线性无关生成元为基
- **定理 1.1.1**: 对任意矩阵 $\boldsymbol{A}$，恒有 $\mathcal{M}(\boldsymbol{A}) = \mathcal{M}(\boldsymbol{A}\boldsymbol{A}')$

### 内积、正交与正交矩阵
- **内积与模**: $(\boldsymbol{a}, \boldsymbol{b}) = \boldsymbol{a}'\boldsymbol{b}$，$\|\boldsymbol{a}\| = (\boldsymbol{a}'\boldsymbol{a})^{1/2}$，标准化 $\boldsymbol{b} = \boldsymbol{a}/\|\boldsymbol{a}\|$
- **正交补空间 ($\mathcal{S}^\perp$)**: $\mathcal{S}^\perp = \{\boldsymbol{x} : \boldsymbol{x} \perp \mathcal{S}\}$
- **正交矩阵 ($\boldsymbol{P}$)**: 方阵满足 $\boldsymbol{P}'\boldsymbol{P} = \boldsymbol{P}\boldsymbol{P}' = \boldsymbol{I}_n$，即 $\boldsymbol{P}^{-1} = \boldsymbol{P}'$

### 对称矩阵与正定性
- **对称矩阵**: $\boldsymbol{A}' = \boldsymbol{A}$，实对称阵不同特征值对应的特征向量正交
- **正定矩阵 ($\boldsymbol{A} > \boldsymbol{0}$)**: 对 $\forall \boldsymbol{x} \ne \boldsymbol{0}$ 有 $\boldsymbol{x}'\boldsymbol{A}\boldsymbol{x} > 0 \Leftrightarrow$ 特征值全 $>0 \Leftrightarrow \boldsymbol{A} = \boldsymbol{B}\boldsymbol{B}'$（$\boldsymbol{B}$ 可逆）
- **半正定矩阵 ($\boldsymbol{A} \ge \boldsymbol{0}$)**: 对 $\forall \boldsymbol{x} \ne \boldsymbol{0}$ 有 $\boldsymbol{x}'\boldsymbol{A}\boldsymbol{x} \ge 0 \Leftrightarrow$ 特征值全 $\ge 0 \Leftrightarrow \boldsymbol{A} = \boldsymbol{B}_{n\times r}\boldsymbol{B}'$（列满秩）

## 02 Kronecker 积与拉直运算
### Kronecker 积 ($\boldsymbol{A} \otimes \boldsymbol{B}$)
- **定义**: $\boldsymbol{A}_{m\times n} \otimes \boldsymbol{B}_{p\times q} = (a_{ij}\boldsymbol{B})_{mp \times nq}$
- **代数性质**: 满足结合律、分配律、数乘 $(\alpha\boldsymbol{A})\otimes(\beta\boldsymbol{B}) = \alpha\beta(\boldsymbol{A}\otimes\boldsymbol{B})$
- **混合乘积**: $(\boldsymbol{A}_1\otimes\boldsymbol{B}_1)(\boldsymbol{A}_2\otimes\boldsymbol{B}_2) = (\boldsymbol{A}_1\boldsymbol{A}_2)\otimes(\boldsymbol{B}_1\boldsymbol{B}_2)$
- **转置与逆**: $(\boldsymbol{A}\otimes\boldsymbol{B})' = \boldsymbol{A}'\otimes\boldsymbol{B}'$，$(\boldsymbol{A}\otimes\boldsymbol{B})^{-1} = \boldsymbol{A}^{-1}\otimes\boldsymbol{B}^{-1}$
- **迹、行列式与秩**: $\operatorname{tr}(\boldsymbol{A}\otimes\boldsymbol{B}) = \operatorname{tr}(\boldsymbol{A})\operatorname{tr}(\boldsymbol{B})$，$|\boldsymbol{A}_m\otimes\boldsymbol{B}_n| = |\boldsymbol{A}|^n|\boldsymbol{B}|^m$，$\operatorname{rank}(\boldsymbol{A}\otimes\boldsymbol{B}) = \operatorname{rank}(\boldsymbol{A})\operatorname{rank}(\boldsymbol{B})$
- **特征值与特征向量**: 若 $\boldsymbol{A}\boldsymbol{x}_i = \lambda_i\boldsymbol{x}_i, \boldsymbol{B}\boldsymbol{y}_j = \mu_j\boldsymbol{y}_j$，则 $\boldsymbol{A}\otimes\boldsymbol{B}$ 特征值为 $\lambda_i\mu_j$，特征向量为 $\boldsymbol{x}_i\otimes\boldsymbol{y}_j$

### 拉直运算 ($\operatorname{vec}(\boldsymbol{A})$)
- **定义**: 按列拉直 $\operatorname{vec}(\boldsymbol{A}) = (\boldsymbol{a}_1', \dots, \boldsymbol{a}_n')'$；按行拉直 $\operatorname{rvec}(\boldsymbol{A}) = (\operatorname{vec}(\boldsymbol{A}'))'$
- **线性与迹关系**: $\operatorname{tr}(\boldsymbol{A}\boldsymbol{B}) = (\operatorname{vec}(\boldsymbol{A}'))'\operatorname{vec}(\boldsymbol{B})$，$\operatorname{vec}(\boldsymbol{a}\boldsymbol{b}') = \boldsymbol{b}\otimes\boldsymbol{a}$
- **核心恒等式**: $\operatorname{vec}(\boldsymbol{A}\boldsymbol{B}\boldsymbol{C}) = (\boldsymbol{C}'\otimes\boldsymbol{A})\operatorname{vec}(\boldsymbol{B})$

## 03 矩阵分解
### QR 分解 ($m \ge n$)
- **Reduced QR**: $\boldsymbol{A}_{m\times n} = \boldsymbol{Q}_{m\times n}\boldsymbol{R}_{n\times n}$（$\boldsymbol{Q}'\boldsymbol{Q}=\boldsymbol{I}_n$，$\boldsymbol{R}$ 上三角；列满秩且 $r_{jj}>0$ 时唯一）
- **Full QR**: $\boldsymbol{A}_{m\times n} = \boldsymbol{Q}_{m\times m}\begin{pmatrix}\boldsymbol{R}\\\boldsymbol{0}\end{pmatrix}_{m\times n}$（$\boldsymbol{Q}$ 为 $m\times m$ 正交阵）
- **非满秩情形**: $\boldsymbol{A}\boldsymbol{P} = \widetilde{\boldsymbol{Q}}_{m\times l}[\boldsymbol{R}_{11}\ \boldsymbol{R}_{12}]$（$\boldsymbol{P}$ 为置换阵，$l = \operatorname{rank}(\boldsymbol{A})$）
- **计算方法**: Classical/Modified Gram-Schmidt ($O(2mn^2)$)、Householder（数值最稳）、Givens（适合稀疏/并行）
- **解线性方程组**: $m \ge n$ 时 $\boldsymbol{x} = \boldsymbol{R}_1^{-1}(\boldsymbol{Q}_1'\boldsymbol{b})$；$m < n$ 时对 $\boldsymbol{A}'$ 作 QR 得 $\boldsymbol{x} = \boldsymbol{Q}\begin{pmatrix}(\boldsymbol{R}_1')^{-1}\boldsymbol{b}\\\boldsymbol{0}\end{pmatrix}$；R 中用 `qr(A)`

### Cholesky、谱分解与 SVD
- **Cholesky 分解**: 若 $\boldsymbol{A} > \boldsymbol{0}$，存在唯一主对角元为正的下三角阵 $\boldsymbol{L}$ 使 $\boldsymbol{A} = \boldsymbol{L}\boldsymbol{L}'$；R 中用 `chol(A)`
- **谱分解**: 对称阵 $\boldsymbol{A} = \boldsymbol{P}\boldsymbol{\Lambda}_n\boldsymbol{P}' = \sum_{i=1}^n \lambda_i\boldsymbol{\psi}_i\boldsymbol{\psi}_i'$；满秩谱分解 $\boldsymbol{A} = \boldsymbol{P}_1\boldsymbol{\Lambda}_r\boldsymbol{P}_1'$
- **奇异值分解 (SVD)**: $\boldsymbol{A}_{m\times n} = \boldsymbol{U}_1\boldsymbol{\Lambda}_r\boldsymbol{V}_1'$（$\lambda_i>0$ 为奇异值，$\lambda_i^2$ 为 $\boldsymbol{A}'\boldsymbol{A}$ 非零特征值）；R 中用 `svd(A)`

## 04 广义逆与正交投影
### 幂等阵与正交投影矩阵
- **幂等阵**: $\boldsymbol{A}^2 = \boldsymbol{A}$，特征值只能为 $0$ 或 $1$，且 $\operatorname{tr}(\boldsymbol{A}) = \operatorname{rank}(\boldsymbol{A})$
- **正交投影阵 ($\boldsymbol{P}_A$)**: $\boldsymbol{P}_A$ 为正交投影阵 $\Leftrightarrow \boldsymbol{P}_A$ 为对称幂等阵；$\boldsymbol{I}-\boldsymbol{P}_A$ 投影到正交补空间 $\mathcal{M}(\boldsymbol{A})^\perp$
- **构造公式**: $\boldsymbol{P}_A = \boldsymbol{A}(\boldsymbol{A}'\boldsymbol{A})^-\boldsymbol{A}'$（与 $(\boldsymbol{A}'\boldsymbol{A})^-$ 取法无关；列满秩时为 $\boldsymbol{A}(\boldsymbol{A}'\boldsymbol{A})^{-1}\boldsymbol{A}'$）

### 减号广义逆 ($\boldsymbol{A}^-$)
- **定义**: 满足 $\boldsymbol{A}\boldsymbol{X}\boldsymbol{A} = \boldsymbol{A}$，存在但不唯一
- **性质**: $\operatorname{rank}(\boldsymbol{A}) = \operatorname{rank}(\boldsymbol{A}\boldsymbol{A}^-) = \operatorname{tr}(\boldsymbol{A}\boldsymbol{A}^-)$
- **相容方程组通解**: $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}$ 通解为 $\boldsymbol{x} = \boldsymbol{A}^-\boldsymbol{b} + (\boldsymbol{I} - \boldsymbol{A}^-\boldsymbol{A})\boldsymbol{z}$（$\boldsymbol{z}$ 任意）

### Moore-Penrose 广义逆 ($\boldsymbol{A}^+$)
- **四个 Penrose 条件**: $\boldsymbol{A}\boldsymbol{X}\boldsymbol{A}=\boldsymbol{A}$、$\boldsymbol{X}\boldsymbol{A}\boldsymbol{X}=\boldsymbol{X}$、$(\boldsymbol{A}\boldsymbol{X})'=\boldsymbol{A}\boldsymbol{X}$、$(\boldsymbol{X}\boldsymbol{A})'=\boldsymbol{X}\boldsymbol{A}$，存在且唯一
- **SVD 构造**: $\boldsymbol{A}^+ = \boldsymbol{V}\begin{pmatrix}\boldsymbol{\Lambda}_r^{-1}&\boldsymbol{0}\\\boldsymbol{0}&\boldsymbol{0}\end{pmatrix}\boldsymbol{U}'$；对称阵 $\boldsymbol{A}=\boldsymbol{H}'\boldsymbol{\Lambda}\boldsymbol{H}$ 时 $\boldsymbol{A}^+=\boldsymbol{H}'\boldsymbol{\Lambda}^+\boldsymbol{H}$
- **特例**: 可逆时 $\boldsymbol{A}^+=\boldsymbol{A}^{-1}$；列满秩 $\boldsymbol{A}^+=(\boldsymbol{A}'\boldsymbol{A})^{-1}\boldsymbol{A}'$（左逆）；行满秩 $\boldsymbol{A}^+=\boldsymbol{A}'(\boldsymbol{A}\boldsymbol{A}')^{-1}$（右逆）
- **常用性质**: $(\boldsymbol{A}^+)^+=\boldsymbol{A}$，$\boldsymbol{A}^+=\boldsymbol{A}'(\boldsymbol{A}\boldsymbol{A}')^+=(\boldsymbol{A}'\boldsymbol{A})^+\boldsymbol{A}'$，$\boldsymbol{A}^+\boldsymbol{A}$ 与 $\boldsymbol{A}\boldsymbol{A}^+$ 均为投影阵

## 05 分块矩阵
### 行列式与 Schur 补
- **分块行列式**: $|\boldsymbol{A}_{11}|\ne 0 \Rightarrow |\boldsymbol{A}| = |\boldsymbol{A}_{11}||\boldsymbol{A}_{22\cdot 1}|$，其中 Schur 补 $\boldsymbol{A}_{22\cdot 1} = \boldsymbol{A}_{22} - \boldsymbol{A}_{21}\boldsymbol{A}_{11}^{-1}\boldsymbol{A}_{12}$
- **对称形式**: $|\boldsymbol{A}_{22}|\ne 0 \Rightarrow |\boldsymbol{A}| = |\boldsymbol{A}_{11\cdot 2}||\boldsymbol{A}_{22}|$，其中 $\boldsymbol{A}_{11\cdot 2} = \boldsymbol{A}_{11} - \boldsymbol{A}_{12}\boldsymbol{A}_{22}^{-1}\boldsymbol{A}_{21}$

### 分块求逆公式
- **Schur 补求逆**: 当 $|\boldsymbol{A}_{11}|\ne 0, |\boldsymbol{A}_{22}|\ne 0$ 时，$\boldsymbol{A}^{-1} = \begin{pmatrix}\boldsymbol{A}_{11\cdot 2}^{-1} & -\boldsymbol{A}_{11}^{-1}\boldsymbol{A}_{12}\boldsymbol{A}_{22\cdot 1}^{-1} \\ -\boldsymbol{A}_{22}^{-1}\boldsymbol{A}_{21}\boldsymbol{A}_{11\cdot 2}^{-1} & \boldsymbol{A}_{22\cdot 1}^{-1}\end{pmatrix}$

## 06 微商与变换的 Jacobi
### 矩阵对标量的微商 ($\partial\boldsymbol{Y}/\partial t$)
- **基本法则**: 线性、乘积法则 $\partial(\boldsymbol{X}\boldsymbol{Y})/\partial t = \frac{\partial\boldsymbol{X}}{\partial t}\boldsymbol{Y} + \boldsymbol{X}\frac{\partial\boldsymbol{Y}}{\partial t}$、逆矩阵 $\partial\boldsymbol{X}^{-1}/\partial t = -\boldsymbol{X}^{-1}\frac{\partial\boldsymbol{X}}{\partial t}\boldsymbol{X}^{-1}$

### 标量函数对矩阵/向量的微商 ($\partial y/\partial\boldsymbol{X}$)
- **向量微商**: $\partial(\boldsymbol{a}'\boldsymbol{x})/\partial\boldsymbol{x} = \boldsymbol{a}$，$\partial(\boldsymbol{x}'\boldsymbol{A}\boldsymbol{x})/\partial\boldsymbol{x} = (\boldsymbol{A}+\boldsymbol{A}')\boldsymbol{x}$
- **迹的微商**: $\partial\operatorname{tr}(\boldsymbol{A}\boldsymbol{X}\boldsymbol{B})/\partial\boldsymbol{X} = \boldsymbol{A}'\boldsymbol{B}'$，$\partial\operatorname{tr}(\boldsymbol{X}'\boldsymbol{A}\boldsymbol{X})/\partial\boldsymbol{X} = (\boldsymbol{A}+\boldsymbol{A}')\boldsymbol{X}$
- **行列式微商**: $\partial\det(\boldsymbol{X})/\partial\boldsymbol{X} = (\boldsymbol{X}')^{-1}\det(\boldsymbol{X})$，$\partial\operatorname{tr}(\boldsymbol{X}^{-1}\boldsymbol{A})/\partial\boldsymbol{X} = -(\boldsymbol{X}^{-1}\boldsymbol{A}\boldsymbol{X}^{-1})'$

### 矩阵对矩阵微商与链式法则
- **定义**: $\partial\boldsymbol{Y}/\partial\boldsymbol{X} = \partial(\operatorname{vec}(\boldsymbol{Y}))'/\partial\operatorname{vec}(\boldsymbol{X})$
- **常用公式**: $\partial(\boldsymbol{A}\boldsymbol{X}\boldsymbol{B})/\partial\boldsymbol{X} = \boldsymbol{B}\otimes\boldsymbol{A}'$，$\partial\boldsymbol{X}^{-1}/\partial\boldsymbol{X} = -\boldsymbol{X}^{-1}\otimes(\boldsymbol{X}^{-1})'$
- **链式法则**: $\partial\psi(\boldsymbol{X})/\partial t = \operatorname{tr}\left[\frac{\partial\boldsymbol{X}}{\partial t}\left(\frac{\partial\psi}{\partial\boldsymbol{X}}\right)'\right]$；$\partial\ln\det(\boldsymbol{X})/\partial t = \operatorname{tr}\left(\boldsymbol{X}^{-1}\frac{\partial\boldsymbol{X}}{\partial t}\right)$

### 变换的 Jacobi 行列式 ($J(\boldsymbol{Y}\to\boldsymbol{X}) = |\partial\boldsymbol{Y}/\partial\boldsymbol{X}|_+$)
- **双线性变换**: $\boldsymbol{Y}_{m\times n} = \boldsymbol{A}\boldsymbol{X}\boldsymbol{B} \Rightarrow J(\boldsymbol{Y}\to\boldsymbol{X}) = |\boldsymbol{A}|_+^n|\boldsymbol{B}|_+^m$；向量 $\boldsymbol{y}=\boldsymbol{A}\boldsymbol{x} \Rightarrow J = |\boldsymbol{A}|_+$
- **矩阵求逆变换**: $\boldsymbol{Y} = \boldsymbol{X}_{n\times n}^{-1} \Rightarrow J(\boldsymbol{Y}\to\boldsymbol{X}) = |\boldsymbol{X}|_+^{-2n}$
