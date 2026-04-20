# 流形几何核心算法与公式

> 作者: math-researcher (ClawTeam manifold-vector)
> 日期: 2026-04-20
> 状态: 完成

---

## 目录

1. [黎曼度量 (Riemannian Metric)](#1-黎曼度量-riemannian-metric)
2. [测地线 (Geodesic)](#2-测地线-geodesic)
3. [曲率 (Curvature)](#3-曲率-curvature)
4. [双曲空间距离公式](#4-双曲空间距离公式)
5. [流形学习算法](#5-流形学习算法)
6. [数值计算示例](#6-数值计算示例)
7. [参考文献](#7-参考文献)

---

## 1. 黎曼度量 (Riemannian Metric)

### 1.1 定义

黎曼度量是流形上定义的**光滑正定对称双线性形式**，它为每个切空间提供了内积结构。

**形式化定义**：

设 $M$ 为 $n$ 维光滑流形，黎曼度量 $g$ 是一个 $(0,2)$ 型张量场，满足：

$$
g: T_p M \times T_p M \to \mathbb{R}
$$

满足以下性质：
- **对称性**: $g(X, Y) = g(Y, X)$
- **正定性**: $g(X, X) > 0$ 当 $X \neq 0$
- **光滑性**: 对光滑向量场 $X, Y$，$g(X, Y)$ 是光滑函数

### 1.2 局部坐标表示

在局部坐标系 $(x^1, \ldots, x^n)$ 下，度量可写为：

$$
g = g_{\mu\nu}(x) \, dx^\mu \otimes dx^\nu
$$

或等价的线元素形式：

$$
ds^2 = g_{\mu\nu}(x) \, dx^\mu dx^\nu
$$

**参数说明**：
- $g_{\mu\nu}$: 度量张量的分量矩阵
- $dx^\mu, dx^\nu$: 坐标微分
- $ds^2$: 无穷小距离的平方

### 1.3 度量分量提取

度量分量可通过评估基向量得到：

$$
g_{\mu\nu}(x) = g\left(\frac{\partial}{\partial x^\mu}, \frac{\partial}{\partial x^\nu}\right)
$$

### 1.4 常见度量示例

#### 欧氏空间 ($\mathbb{R}^n$)

$$
g_{\mu\nu} = \delta_{\mu\nu} = \begin{cases} 1 & \mu = \nu \\ 0 & \mu \neq \nu \end{cases}
$$

线元素：
$$
ds^2 = dx_1^2 + dx_2^2 + \cdots + dx_n^2
$$

#### 球面 ($S^2$)

极坐标 $(\theta, \phi)$ 下的度量：

$$
ds^2 = d\theta^2 + \sin^2\theta \, d\phi^2
$$

度量矩阵：
$$
g = \begin{pmatrix} 1 & 0 \\ 0 & \sin^2\theta \end{pmatrix}
$$

#### Poincaré 盘模型 (双曲空间)

单位盘 $D = \{z \in \mathbb{C} : |z| < 1\}$ 上的度量：

$$
ds^2 = \frac{4(dx^2 + dy^2)}{(1 - x^2 - y^2)^2}
$$

或等价地用复坐标：

$$
ds^2 = \frac{4 |dz|^2}{(1 - |z|^2)^2}
$$

### 1.5 度量的作用

1. **向量长度**：
$$
|X| = \sqrt{g(X, X)} = \sqrt{g_{\mu\nu} X^\mu X^\nu}
$$

2. **向量间角度**：
$$
\cos\theta = \frac{g(X, Y)}{|X||Y|}
$$

3. **曲线长度**：
$$
L(\gamma) = \int_a^b \sqrt{g_{\mu\nu} \frac{dx^\mu}{dt} \frac{dx^\nu}{dt}} \, dt
$$

4. **两点距离**：
$$
d(p, q) = \inf_\gamma L(\gamma)
$$
（取所有连接 $p, q$ 的曲线的最小长度）

---

## 2. 测地线 (Geodesic)

### 2.1 定义

测地线是流形上的"最短路径"，是欧氏空间中直线的推广。

**形式化定义**：

测地线 $\gamma: [a, b] \to M$ 是满足测地线方程的曲线：

$$
\frac{d^2 x^\mu}{dt^2} + \Gamma^\mu_{\alpha\beta} \frac{dx^\alpha}{dt} \frac{dx^\beta}{dt} = 0
$$

### 2.2 Christoffel 符号

Christoffel 符号是度量导出的联络系数：

$$
\Gamma^\mu_{\alpha\beta} = \frac{1}{2} g^{\mu\lambda} \left( \frac{\partial g_{\lambda\alpha}}{\partial x^\beta} + \frac{\partial g_{\lambda\beta}}{\partial x^\alpha} - \frac{\partial g_{\alpha\beta}}{\partial x^\lambda} \right)
$$

**参数说明**：
- $\Gamma^\mu_{\alpha\beta}$: 第二类 Christoffel 符号
- $g^{\mu\lambda}$: 度量张量的逆矩阵元素
- 前两个下标对称：$\Gamma^\mu_{\alpha\beta} = \Gamma^\mu_{\beta\alpha}$

### 2.3 测地线方程推导

从变分原理出发，最小化弧长泛函：

$$
S = \int_a^b \sqrt{g_{\mu\nu} \frac{dx^\mu}{dt} \frac{dx^\nu}{dt}} \, dt
$$

使用 Euler-Lagrange 方程：

$$
\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{x}^\gamma}\right) - \frac{\partial L}{\partial x^\gamma} = 0
$$

其中 $L = \sqrt{g_{\mu\nu} \dot{x}^\mu \dot{x}^\nu}$。

经过推导得到测地线方程。

### 2.4 测地线性质

1. **局部最短性**: 测地线在局部是最短路径
2. **唯一性**: 给定起点和初始方向，测地线唯一确定
3. **自平行性**: 测地线的切向量沿曲线自身平行移动不变

### 2.5 典型测地线示例

#### 欧氏空间
所有 Christoffel 符号为零，测地线方程：
$$
\frac{d^2 x^\mu}{dt^2} = 0
$$
解为直线：$x^\mu(t) = a^\mu t + b^\mu$

#### 球面 ($S^2$)
测地线是**大圆**（过球心的平面与球面的交线）

#### Poincaré 盘
测地线是：
- 过原心的直线（直径）
- 与边界圆正交的圆弧

---

## 3. 曲率 (Curvature)

### 3.1 Gaussian 曲率

**定义**（2维曲面）：

Gaussian 曲率是两个主曲率的乘积：

$$
K = k_1 \cdot k_2
$$

**通过度量计算**：

$$
K = \frac{R_{1212}}{g_{11}g_{22} - g_{12}^2}
$$

其中 $R_{1212}$ 是 Riemann 曲率张量分量。

### 3.2 三种曲率类型

| 曲率类型 | 值 | 示例 | 几何特征 |
|---------|---|------|---------|
| **正曲率** | $K > 0$ | 球面 | 三角形内角和 > 180° |
| **零曲率** | $K = 0$ | 平面、圆柱 | 三角形内角和 = 180° |
| **负曲率** | $K < 0$ | 双曲空间 | 三角形内角和 < 180° |

### 3.3 Riemann 曲率张量

**定义**：

$$
R(X, Y)Z = \nabla_X \nabla_Y Z - \nabla_Y \nabla_X Z - \nabla_{[X,Y]} Z
$$

**分量表示**：

$$
R^\rho_{\sigma\mu\nu} = \frac{\partial \Gamma^\rho_{\nu\sigma}}{\partial x^\mu} - \frac{\partial \Gamma^\rho_{\mu\sigma}}{\partial x^\nu} + \Gamma^\rho_{\mu\lambda}\Gamma^\lambda_{\nu\sigma} - \Gamma^\rho_{\nu\lambda}\Gamma^\lambda_{\mu\sigma}
$$

**性质**：
- 对称性：$R_{\sigma\mu\nu\rho} = -R_{\sigma\nu\mu\rho}$
- Bianchi 恒等式：$R_{\sigma\mu\nu\rho} + R_{\sigma\nu\rho\mu} + R_{\sigma\rho\mu\nu} = 0$

### 3.4 截面曲率

**定义**：

对于切平面 $\Pi = \text{span}(u, v)$：

$$
K(u, v) = \frac{\langle R(u, v)v, u \rangle}{\langle u, u \rangle \langle v, v \rangle - \langle u, v \rangle^2}
$$

**意义**：
- 截面曲率是 Gaussian 曲率的高维推广
- 描述流形在特定切平面方向的弯曲程度

### 3.5 Ricci 曲率

**定义**：

$$
R_{\mu\nu} = R^\lambda_{\mu\lambda\nu} = \sum_\lambda R^\lambda_{\mu\lambda\nu}
$$

**性质**：
- Ricci 曲率是 Riemann 曲率张量的缩并
- Einstein 场方程中使用 Ricci 曲率

### 3.6 标量曲率

$$
S = g^{\mu\nu} R_{\mu\nu} = \sum_{\mu,\nu} g^{\mu\nu} R_{\mu\nu}
$$

---

## 4. 双曲空间距离公式

### 4.1 Poincaré 球模型

#### 4.1.1 度量

$n$ 维 Poincaré 球模型：
$$
\mathbb{D}^n = \{x \in \mathbb{R}^n : \|x\| < 1\}
$$

度量张量：
$$
g_{\mu\nu} = \frac{4\delta_{\mu\nu}}{(1 - \|x\|^2)^2}
$$

#### 4.1.2 距离公式

两点 $x, y \in \mathbb{D}^n$ 之间的双曲距离：

$$
d_{\mathbb{D}}(x, y) = \text{arccosh}\left(1 + 2\frac{\|x - y\|^2}{(1 - \|x\|^2)(1 - \|y\|^2)}\right)
$$

**参数说明**：
- $\|x\|$: 向量 $x$ 的欧氏范数
- $\|x - y\|$: 两点的欧氏距离
- $\text{arccosh}$: 反双曲余弦函数

**简化形式**（定义 $\alpha = \|x\|^2$, $\beta = \|y\|^2$）：

$$
d_{\mathbb{D}}(x, y) = \text{arccosh}\left(\frac{1 - \alpha\beta + \|x - y\|^2}{(1 - \alpha)(1 - \beta)}\right)
$$

#### 4.1.3 与原点的距离

点 $x$ 到原点 $O$ 的距离：

$$
d_{\mathbb{D}}(O, x) = \text{arctanh}(\|x\|) = \frac{1}{2}\ln\left(\frac{1 + \|x\|}{1 - \|x\|}\right)
$$

### 4.2 Lorentz 模型 (Hyperboloid 模型)

#### 4.2.1 定义

Lorentz 模型中的点是前向双曲面上的点：

$$
\mathbb{H}^n = \{x \in \mathbb{R}^{n+1} : x_0 > 0, \langle x, x \rangle_L = -1\}
$$

其中 Lorentz 内积：
$$
\langle x, y \rangle_L = -x_0 y_0 + \sum_{i=1}^n x_i y_i
$$

#### 4.2.2 距离公式

两点 $x, y \in \mathbb{H}^n$ 之间的双曲距离：

$$
d_{\mathbb{H}}(x, y) = \text{arccosh}(-\langle x, y \rangle_L)
$$

或展开：

$$
d_{\mathbb{H}}(x, y) = \text{arccosh}(x_0 y_0 - \sum_{i=1}^n x_i y_i)
$$

#### 4.2.3 Lorentz 距平方形式

实践中常用**平方 Lorentz 距离**：

$$
d_L^2(x, y) = -2 - 2\langle x, y \rangle_L = \|x - y\|_L^2
$$

优势：
- 计算更高效
- 避免 arccosh 函数
- 优化时梯度更稳定

### 4.3 模型转换

#### Poincaré ↔ Lorentz

从 Lorentz 到 Poincaré：
$$
x_{\mathbb{D}} = \frac{(x_1, \ldots, x_n)}{x_0 + 1}
$$

从 Poincaré 到 Lorentz：
$$
x_{\mathbb{H}} = \frac{(1 + \|x_{\mathbb{D}}\|^2, 2x_{\mathbb{D}})}{1 - \|x_{\mathbb{D}}\|^2}
$$

### 4.4 负曲率的意义

双曲空间具有**常数负曲率** $K = -1$：

| 特性 | 欧氏空间 ($K=0$) | 双曲空间 ($K=-1$) |
|-----|----------------|------------------|
| 平行线 | 唯一一条 | 无限多条 |
| 三角形内角和 | 180° | < 180° |
| 圆周长-半径关系 | $C = 2\pi r$ | $C \propto e^r$ |
| 圆面积-半径关系 | $A = \pi r^2$ | $A \propto e^{2r}$ |

---

## 5. 流形学习算法

### 5.1 Isomap (Isometric Mapping)

#### 5.1.1 核心思想

Isomap 通过保持**测地距离**来实现非线性降维。

#### 5.1.2 算法流程

1. **构建邻域图**：
   - k-NN 或 $\epsilon$-ball 邻域
   - 图边权重 = 欧氏距离

2. **计算测地距离**：
   - 使用 Dijkstra 或 Floyd 算法
   - 图上最短路径近似测地距离

3. **经典 MDS 嵌入**：
   - 将测地距离矩阵转换为低维坐标

#### 5.1.3 数学公式

**测地距离矩阵** $D_G$：
$$
D_G(i, j) = \min_{\text{path } P} \sum_{(p, q) \in P} d(p, q)
$$

**经典 MDS**：

给定距离矩阵 $D$，计算 Gram 矩阵：
$$
B = -\frac{1}{2} H D^2 H
$$

其中 $H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^T$ 是中心化矩阵。

求 $B$ 的前 $d$ 个最大特征值 $\lambda_1, \ldots, \lambda_d$ 和对应特征向量 $v_1, \ldots, v_d$：

低维嵌入：
$$
Y = [\sqrt{\lambda_1} v_1, \ldots, \sqrt{\lambda_d} v_d]
$$

### 5.2 LLE (Locally Linear Embedding)

#### 5.2.1 核心思想

假设数据在局部是线性的，每个点可由邻居线性重构。

#### 5.2.2 算法流程

1. **寻找邻居**

2. **计算重构权重**：
   最小化重构误差：
   $$
   \min_W \sum_i \|x_i - \sum_{j \in N(i)} W_{ij} x_j\|^2
   $$
   约束：$\sum_j w_{ij} = 1$

3. **计算低维嵌入**：
   保持权重不变，最小化：
   $$
   \min_Y \sum_i \|y_i - \sum_{j \in N(i)} w_{ij} y_j\|^2
   $$

#### 5.2.3 权重求解

局部协方差矩阵：
$$
C_i = (X_i - x_i)(X_i - x_i)^T
$$

权重解：
$$
w_i = \frac{C_i^{-1} \mathbf{1}}{\mathbf{1}^T C_i^{-1} \mathbf{1}}
$$

#### 5.2.4 嵌入求解

构建矩阵：
$$
M = (I - W)^T (I - W)
$$

低维坐标是 $M$ 的最小 $d+1$ 个特征向量（排除最小的一个）。

### 5.3 t-SNE (t-Distributed Stochastic Neighbor Embedding)

#### 5.3.1 核心思想

将高维空间的相似性转换为概率，低维空间使用 t 分布匹配。

#### 5.3.2 高维相似性

**对称化 pairwise 相似性**：

高维点 $i, j$ 的相似概率：
$$
p_{ij} = \frac{p_{i|j} + p_{j|i}}{2n}
$$

其中：
$$
p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}
$$

**参数说明**：
- $\sigma_i$: 点 $i$ 的局部带宽（由 perplexity 参数控制）
- perplexity 通常设为 5-50

#### 5.3.3 低维相似性

低维空间使用**学生 t 分布**（自由度=1，即 Cauchy 分布）：

$$
q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}
$$

#### 5.3.4 优化目标

最小化 KL 散度：
$$
C = KL(P|Q) = \sum_{i,j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
$$

#### 5.3.5 梯度

$$
\frac{\partial C}{\partial y_i} = 4 \sum_j (p_{ij} - q_{ij})(y_i - y_j)(1 + \|y_i - y_j\|^2)^{-1}
$$

### 5.4 算法对比

| 算法 | 保持性质 | 优点 | 缺点 |
|-----|---------|------|------|
| **Isomap** | 测地距离 | 全局结构 | 计算量大 |
| **LLE** | 局部线性 | 快速 | 对噪声敏感 |
| **t-SNE** | 局部概率 | 可视化好 | 不保持距离 |

---

## 6. 数值计算示例

### 6.1 Poincaré 盘距离计算

**问题**：计算 $\mathbb{D}^2$ 中 $x = (0.3, 0.2)$ 和 $y = (0.5, 0.1)$ 的双曲距离。

**步骤**：

1. 计算各点范数：
   - $\|x\|^2 = 0.3^2 + 0.2^2 = 0.13$
   - $\|y\|^2 = 0.5^2 + 0.1^2 = 0.26$

2. 计算欧氏距离平方：
   - $\|x - y\|^2 = (0.3-0.5)^2 + (0.2-0.1)^2 = 0.04 + 0.01 = 0.05$

3. 代入公式：
   $$
   d = \text{arccosh}\left(1 + 2\frac{0.05}{(1-0.13)(1-0.26)}\right)
   $$
   $$
   = \text{arccosh}\left(1 + \frac{0.1}{0.87 \times 0.74}\right)
   $$
   $$
   = \text{arccosh}\left(1 + \frac{0.1}{0.6438}\right)
   $$
   $$
   = \text{arccosh}(1 + 0.1553) = \text{arccosh}(1.1553)
   $$
   $$
   \approx 0.568
   $$

### 6.2 Lorentz 模型距离计算

**问题**：计算 $\mathbb{H}^2$ 中 $x = (2.0, 1.0, \sqrt{2})$ 和 $y = (3.0, 2.0, \sqrt{5})$ 的距离。

**验证点在双曲面上**：
- $\langle x, x \rangle_L = -4 + 1 + 2 = -1$ ✓
- $\langle y, y \rangle_L = -9 + 4 + 5 = -1$ ✓

**计算距离**：
$$
\langle x, y \rangle_L = -2.0 \times 3.0 + 1.0 \times 2.0 + \sqrt{2} \times \sqrt{5}
$$
$$
= -6 + 2 + \sqrt{10} = -4 + 3.162 = -0.838
$$

$$
d = \text{arccosh}(-(-0.838)) = \text{arccosh}(0.838)
$$

注意：这里 $\langle x, y \rangle_L$ 应为负值（因 $x, y$ 都在前向双曲面上），正确计算：
$$
d = \text{arccosh}(-\langle x, y \rangle_L)
$$

### 6.3 球面 Christoffel 符号计算

**度量**：
$$
g = \begin{pmatrix} 1 & 0 \\ 0 & \sin^2\theta \end{pmatrix}
$$
$$
g^{-1} = \begin{pmatrix} 1 & 0 \\ 0 & \sin^{-2}\theta \end{pmatrix}
$$

**计算** $\Gamma^\theta_{\phi\phi}$：

$$
\Gamma^\theta_{\phi\phi} = \frac{1}{2} g^{\theta\theta} \left(-\frac{\partial g_{\phi\phi}}{\partial \theta}\right)
$$
$$
= \frac{1}{2} \cdot 1 \cdot (-2\sin\theta\cos\theta) = -\sin\theta\cos\theta
$$

**计算** $\Gamma^\phi_{\theta\phi}$：

$$
\Gamma^\phi_{\theta\phi} = \frac{1}{2} g^{\phi\phi} \frac{\partial g_{\phi\phi}}{\partial \theta}
$$
$$
= \frac{1}{2} \sin^{-2}\theta \cdot 2\sin\theta\cos\theta = \cot\theta
$$

### 6.4 测地线方程数值求解

**球面测地线**（从 $(\theta_0, \phi_0)$出发，初始方向 $(\dot{\theta}_0, \dot{\phi}_0)$）：

测地线方程：
$$
\ddot{\theta} - \sin\theta\cos\theta \dot{\phi}^2 = 0
$$
$$
\ddot{\phi} + 2\cot\theta \dot{\theta}\dot{\phi} = 0
$$

可用数值方法（如 Runge-Kutta）求解。

---

## 7. 参考文献

### 学术论文

1. **Tenenbaum, de Silva, Langford (2000)**: "A Global Geometric Framework for Nonlinear Dimensionality Reduction" - Isomap
2. **Roweis, Saul (2000)**: "Nonlinear Dimensionality Reduction by Locally Linear Embedding" - LLE
3. **van der Maaten, Hinton (2008)**: "Visualizing Data using t-SNE"
4. **Nickel, Kiela (2017)**: "Poincaré Embeddings for Learning Hierarchical Representations"
5. **Law et al. (2019)**: "Lorentzian Distance Learning for Hyperbolic Representations" - Lorentz 模型

### 教材与笔记

- David Tong: "General Relativity" - Riemannian Geometry 章节
- Lee: "Introduction to Riemannian Manifolds"
- Do Carmo: "Riemannian Geometry"

### 网络资源

- Wikipedia: Poincaré disk model, Riemann curvature tensor, Sectional curvature
- scikit-learn 文档: Manifold Learning
- Brian Keng's Blog: "Hyperbolic Geometry and Poincaré Embeddings"

---

## 附录：关键公式速查表

| 公式名称 | 表达式 |
|---------|--------|
| 度量线元素 | $ds^2 = g_{\mu\nu} dx^\mu dx^\nu$ |
| Christoffel 符号 | $\Gamma^\mu_{\alpha\beta} = \frac{1}{2}g^{\mu\lambda}(\partial_\beta g_{\lambda\alpha} + \partial_\alpha g_{\lambda\beta} - \partial_\lambda g_{\alpha\beta})$ |
| 测地线方程 | $\ddot{x}^\mu + \Gamma^\mu_{\alpha\beta}\dot{x}^\alpha\dot{x}^\beta = 0$ |
| Riemann 曲率 | $R^\rho_{\sigma\mu\nu} = \partial_\mu\Gamma^\rho_{\nu\sigma} - \partial_\nu\Gamma^\rho_{\mu\sigma} + \Gamma^\rho_{\mu\lambda}\Gamma^\lambda_{\nu\sigma} - \Gamma^\rho_{\nu\lambda}\Gamma^\lambda_{\mu\sigma}$ |
| 截面曲率 | $K(u,v) = \frac{\langle R(u,v)v,u\rangle}{\langle u,u\rangle\langle v,v\rangle - \langle u,v\rangle^2}$ |
| Poincaré 距离 | $d = \text{arccosh}(1 + 2\frac{\|x-y\|^2}{(1-\|x\|^2)(1-\|y\|^2)})$ |
| Lorentz 距离 | $d = \text{arccosh}(-\langle x,y\rangle_L)$ |
| t-SNE 目标 | $C = \sum_{ij} p_{ij}\log(p_{ij}/q_{ij})$ |

---

*文档完成于 2026-04-20*
*ClawTeam manifold-vector 研究组*