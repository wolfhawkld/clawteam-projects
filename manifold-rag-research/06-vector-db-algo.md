# 向量数据库核心算法

> 作者: vector-researcher (ClawTeam manifold-vector)
> 日期: 2026-04-20
> 状态: 完成

---

## 目录

1. [HNSW (Hierarchical Navigable Small World)](#1-hnsw-hierarchical-navigable-small-world)
2. [FAISS 索引结构](#2-faiss-索引结构)
3. [距离度量](#3-距离度量)
4. [Hub Formation 问题](#4-hub-formation-问题)
5. [高维空间的距离集中现象](#5-高维空间的距离集中现象)
6. [与流形几何的结合点](#6-与流形几何的结合点)
7. [伪代码实现](#7-伪代码实现)
8. [参考文献](#8-参考文献)

---

## 1. HNSW (Hierarchical Navigable Small World)

### 1.1 核心思想

HNSW 是一种**基于图的 ANN 搜索算法**，通过构建**多层级的可导航小世界图**实现快速向量检索。

**设计基础**：
- **跳表 (Skip List)**：多层级结构，高层跳过更多元素，实现 O(log N) 搜索
- **NSW 图 (Navigable Small World)**：随机插入节点，连接最近邻居，形成"小世界"特性

### 1.2 层级结构设计

```
Layer 3 (最高层):  [●]                    ← 入口点，元素最少
                  ↓
Layer 2:         [●]──[●]                 ← 中层，部分元素
                 ↓    ↓
Layer 1:        [●]──[●]──[●]──[●]        ← 较低层，更多元素
               ↓    ↓    ↓    ↓
Layer 0 (底层): [●]──[●]──[●]──[●]──[●]... ← 全部元素，最密集
```

**层级分配规则**：

每个元素随机分配一个层级 $l$，层级由指数分布决定：

$$
P(l) = (1 - m_L^{-1}) \cdot m_L^{-l}
$$

其中 $m_L$ 是层级乘数因子，通常取 $1/\ln M$（M 是每层最大连接数）。

**性质**：
- 高层元素稀疏，用于快速跳转定位区域
- 低层元素密集，用于精确搜索近邻
- 搜索从最高层开始，逐层下潜到目标区域

### 1.3 节点连接选择算法（邻居选择策略）

**插入时的邻居选择**：

当插入新元素 $q$ 时，需要在各层选择连接邻居：

1. **简单选择（简单启发式）**：
   - 选择距离最近的 $M$ 个邻居

2. **启发式选择（推荐）**：
   - 不仅考虑距离，还考虑**连接多样性**
   - 避免所有邻居都在同一方向，导致搜索死角

**启发式选择算法**：

```
Algorithm: SELECT-NEIGHBORS-HEURISTIC(q, C, M)
Input: 新元素q, 候选集C, 最大邻居数M
Output: 选出的M个邻居

1. R ← ∅  (结果集)
2. W ← C  (工作候选集，按距离排序)

3. while |R| < M and W ≠ ∅:
   4. e ← pop nearest element from W
   5. if R = ∅ or distance(q, e) < distance(q, f) for any f ∈ R:
      6.    R ← R ∪ {e}  (添加到结果)
   7. else:
      8.    // 检查e是否与已有邻居方向冲突
      9.    if ∃f ∈ R: distance(e, f) < distance(q, e):
         10.       continue  (e离已有邻居f更近，跳过e)
      11.    else:
         12.       R ← R ∪ {e}

13. return R
```

**直觉解释**：
- 如果新候选 $e$ 离某个已选邻居 $f$ 更近，说明 $e$ 和 $f$ 在同一方向
- 这样的连接冗余，不利于搜索遍历
- 应选择分布在不同方向的邻居

### 1.4 搜索过程（从顶层到底层的导航）

**搜索算法**：

```
Algorithm: HNSW-SEARCH(q, index, ef, K)
Input: 查询向量q, HNSW索引, 搜索宽度ef, 返回数量K
Output: K个最近邻居及其距离

1. // 从最高层开始
2. ep ← get entry point (最高层节点)
3. L ← max layer of ep

4. // 阶段1: 高层快速定位
5. for lc from L down to 1:
   6.    W ← SEARCH-LAYER(q, {ep}, ef, lc)
   7.    ep ← nearest element in W

8. // 阶段2: 底层精确搜索
9. W ← SEARCH-LAYER(q, {ep}, ef, 0)

10. // 返回最近的K个结果
11. return top K elements from W with their distances

---

Algorithm: SEARCH-LAYER(q, entryPoints, ef, layer)
Input: 查询q, 入口点集合, 搜索宽度ef, 层级
Output: ef个候选节点

1. visited ← entryPoints
2. candidates ← entryPoints  (按距离排序，最小堆)
3. results ← entryPoints     (按距离排序，最大堆)

4. while candidates ≠ ∅:
   5.    c ← pop nearest from candidates
   6.    f ← farthest in results
   
   7.    if distance(q, c) > distance(q, f) and |results| = ef:
      8.       break  (所有候选都比结果中最远的还远)
   
   9.    // 遍历c的邻居
   10.    for each neighbor e of c at layer:
      11.       if e not in visited:
         12.          visited ← visited ∪ {e}
         13.          f ← farthest in results
         
         14.          if distance(q, e) < distance(q, f) or |results| < ef:
            15.             candidates ← candidates ∪ {e}
            16.             results ← results ∪ {e}
            17.             if |results| > ef:
               18.                remove farthest from results

19. return results
```

**搜索路径示例**：

```
查询q的搜索路径：

Layer 3:   A → B (找到大致区域)
           ↓
Layer 2:   B → C → D (缩小范围)
           ↓
Layer 1:   D → E → F → G (继续收敛)
           ↓
Layer 0:   G → H → I → J → K (精确搜索)
           
最终返回: {J, K, I} (最近的3个)
```

### 1.5 距离计算在 HNSW 中的使用

**距离的作用**：
1. **邻居选择**：选择距离最近的节点连接
2. **搜索引导**：贪心地向距离更近的邻居移动
3. **结果排序**：返回距离最近的 K 个结果

**优化技巧**：

1. **距离缓存**：
   - 高层搜索时，某些距离可复用
   - 如已计算 dist(q, ep)，下一层可用 ep 作为新入口

2. **提前终止**：
   - 当候选中最近的都比结果中最远的还远，提前终止
   - 避免无效的邻居遍历

3. **批量计算**：
   - 同时计算到多个邻居的距离，利用 SIMD 优化

### 1.6 关键参数

| 参数 | 含义 | 影响 |
|-----|------|------|
| **M** | 每层最大连接数 | M↑ → recall↑, memory↑, build_time↑ |
| **M_max0** | 底层最大连接数（通常 M_max0 = 2M） | 底层连接更多 |
| **ef_construction** | 构建时的搜索宽度 | ef↑ → quality↑, build_time↑ |
| **ef_search** | 搜索时的搜索宽度 | ef↑ → recall↑, search_time↑ |
| **m_L** | 层级乘数因子 | 控制层级分布稀疏度 |

**推荐配置**（根据数据规模）：

| 数据规模 | M | ef_construction | ef_search |
|---------|---|----------------|-----------|
| 小 (<10K) | 16 | 100 | 50 |
| 中 (10K-1M) | 32 | 200 | 100 |
| 大 (>1M) | 48-64 | 400 | 200 |

---

## 2. FAISS 索引结构

### 2.1 FAISS 架构概览

FAISS (Facebook AI Similarity Search) 是一个**向量索引工具箱**，提供多种索引类型的组合使用。

**核心组件**：
- **向量压缩 (Compression)**：减少内存占用
- **非穷尽搜索 (Non-exhaustive Search)**：加速检索
- **精确索引 (Exact Index)**：作为量化器的基准

### 2.2 IVF (Inverted File Index) 结构

**核心思想**：将向量空间划分为多个区域（Voronoi cells），搜索时只访问相关区域。

#### 2.2.1 Voronoi 分割

给定 $n_{list}$ 个中心点（通过 k-means 生成），将空间划分为 $n_{list}$ 个 Voronoi cell：

$$
V_i = \{x \in \mathbb{R}^d : \text{dist}(x, c_i) \leq \text{dist}(x, c_j), \forall j \neq i\}
$$

**结构示意**：

```
        Voronoi 分割示意
        
   ┌─────────────────────────┐
   │   ┌───┐      ┌──────┐   │
   │   │ C1│  ┌───│  C2  │   │
   │   └───┘  │   └──────┘   │
   │          │              │
   │   ┌──────┴───┐  ┌────┐  │
   │   │    C3    │  │ C4 │  │
   │   └──────────┘  └────┘  │
   └─────────────────────────┘
   
C1-C4: Voronoi cell centers
每个cell内存储该区域的向量列表
```

#### 2.2.2 IVF 搜索流程

```
Algorithm: IVF-SEARCH(q, index, n_probe, K)
Input: 查询q, IVF索引, 扫描cell数n_probe, 返回数K
Output: K个最近邻居

1. // 阶段1: 找到最近的n_probe个cell
2. distances_cells ← compute distances to all centroids
3. nearest_cells ← select n_probe cells with smallest distance

4. // 阶段2: 在这些cell内搜索
5. candidates ← ∅
6. for each cell i in nearest_cells:
   7.    candidates ← candidates ∪ vectors in cell i

8. // 阶段3: 精确距离计算
9. results ← compute distances to all candidates
10. return top K from results
```

**复杂度分析**：

| 操作 | 复杂度 |
|-----|--------|
| 找最近的 cell | $O(n_{list} \cdot d)$ |
| 搜索 cell 内向量 | $O(n_{probe} \cdot \bar{n}_{cell} \cdot d)$ |
| 总复杂度 | $O((n_{list} + n_{probe} \cdot \bar{n}_{cell}) \cdot d)$ |

其中 $\bar{n}_{cell} = n / n_{list}$ 是平均每个 cell 的向量数。

#### 2.2.3 IVF 参数选择

**n_list 选择**（数据规模）：

| 数据规模 | 推荐 n_list |
|---------|-------------|
| 10K-100K | $\sqrt{n}$ ~ 100-300 |
| 100K-1M | $\sqrt{n}$ ~ 300-1000 |
| 1M-10M | $4\sqrt{n}$ ~ 4000-12000 |
| >10M | 8192-16384 或更多 |

**经验公式**：
$$
n_{list} \approx \frac{n}{C}
$$
其中 $C$ 是每个 cell 内期望搜索的向量数，通常 20-100。

**n_probe 选择**：
- 小数据集：n_probe = n_list（相当于 flat index）
- 大数据集：n_probe = 1-16，平衡 recall 和速度
- 高 recall：n_probe ↑，但搜索时间也 ↑

### 2.3 PQ (Product Quantization) 压缩

#### 2.3.1 核心原理

**目标**：将高维向量压缩到极小的内存占用（97%压缩率）。

**方法**：
1. 将向量 $x \in \mathbb{R}^D$ 分割为 $m$ 个子向量
2. 每个子向量独立量化为 $k^*$ 个码字
3. 用码字 ID 表示向量

**数学描述**：

给定向量 $x = [x_1, ..., x_D]$，分割为 $m$ 个子向量：

$$
x = [u_1, u_2, ..., u_m]
$$

其中 $u_j \in \mathbb{R}^{D/m}$（假设 $D$ 被 $m$ 整除）。

每个子向量空间有 $k^*$ 个码字（centroid）：

$$
C_j = \{c_{j,1}, c_{j,2}, ..., c_{j,k^*}\}
$$

编码后：

$$
q(x) = [q_1(u_1), q_2(u_2), ..., q_m(u_m)]
$$

其中 $q_j(u_j) = \arg\min_i \|u_j - c_{j,i}\|$。

**码字 ID 范围**：$[0, k^*-1]$，通常 $k^* = 256$（8 bits）。

#### 2.3.2 内存压缩计算

| 参数 | 原始向量 | PQ压缩后 |
|-----|---------|---------|
| 维度 D | 128 | 128 |
| 元素类型 | float32 (32 bits) | uint8 (8 bits) × m |
| m = 8 时 | 128 × 32 = 4096 bits | 8 × 8 = 64 bits |
| **压缩率** | - | **64× (97%)** |

#### 2.3.3 PQ 搜索：非对称距离计算 (ADC)

**非对称距离计算**：查询向量不压缩，数据库向量压缩。

距离近似：

$$
\|q - x\|^2 \approx \|q - D(q(x))\|^2 = \sum_{j=1}^{m} \|u_j^q - c_{j,q_j(x)}\|^2
$$

**预计算查表优化**：

对每个查询 $q$，预先计算到各子空间码字的距离表：

```
Algorithm: PQ-ADC-SEARCH(q, PQ_index, K)
Input: 查询q (不压缩), PQ索引, 返回数K
Output: K个最近邻居

1. // 预计算查表
2. for j in 1..m:
   3.    for i in 0..k^*-1:
      4.       LUT[j][i] = \|u_j^q - c_{j,i}\|^2

5. // 计算到所有编码向量的距离
6. for each encoded vector [id_1, ..., id_m] in index:
   7.    dist = Σ_{j=1}^{m} LUT[j][id_j]  // 仅查表求和
   8.    keep if in top K

9. return top K vectors
```

**复杂度**：
- 查表构建：$O(m \cdot k^* \cdot D/m) = O(k^* \cdot D)$
- 每向量距离：$O(m)$（仅查表求和，不涉及向量运算）
- 总复杂度：$O(k^* \cdot D + n \cdot m)$

### 2.4 IVF-PQ 混合索引

**组合 IVF + PQ**：先用 IVF 缩小搜索范围，再用 PQ 压缩向量。

```
IVF-PQ 结构：

    ┌─────────────────────────┐
    │     Voronoi Cells       │
    │   (n_list 个区域)        │
    └─────────────────────────┘
              │
              ↓ 每个cell内
    ┌─────────────────────────┐
    │   PQ编码的向量列表       │
    │   [id_1,...,id_m]       │
    │   仅占 m×8 bits         │
    └─────────────────────────┘
```

**搜索流程**：

```
Algorithm: IVF-PQ-SEARCH(q, index, n_probe, K)
Input: 查询q, IVF-PQ索引, n_probe, K
Output: K个最近邻居

1. // 找最近的cell
2. nearest_cells ← find n_probe nearest centroids

3. // 预计算PQ查表
4. LUT ← build lookup tables from q

5. // 在cell内搜索PQ向量
6. candidates ← ∅
7. for each cell in nearest_cells:
   8.    for each PQ-encoded vector in cell:
      9.       dist ← Σ LUT[j][id_j]
      10.      candidates ← candidates ∪ (vector_id, dist)

11. return top K from candidates
```

**性能对比**（SIFT1M 数据集）：

| 索引类型 | Recall@100 | 搜索时间 | 内存占用 |
|---------|------------|---------|---------|
| Flat (精确) | 100% | 61.2 ms | 256 MB |
| PQ | 50% | 1.49 ms | 6.5 MB |
| IVF-PQ | 52% (n_probe=48) | 0.09 ms | 9.2 MB |

**IVF-PQ 提供**：
- 92× 搜索加速
- 96% 内存压缩
- 合理的 recall (~50%)

### 2.5 HNSW+IVF 混合索引

**组合 HNSW + IVF**：用 HNSW 作为 IVF 的粗量化器（coarse quantizer）。

**动机**：
- IVF 的 k-means 量化器在高维空间效率低
- HNSW 可快速找到最近的 cell centroid

**结构**：

```
HNSW-IVF 结构：

┌─────────────────────────┐
│  HNSW (coarse quantizer)│ ← 存储所有 centroid
│  多层图结构              │
└─────────────────────────┘
         │
         ↓ 找到最近 centroid
┌─────────────────────────┐
│  Inverted Lists         │ ← 每个 centroid 对应的向量列表
│  (可进一步用 PQ 压缩)    │
└─────────────────────────┘
```

**优势**：
- 高效的 centroid 搜索（O(log n_list) vs O(n_list)）
- 适合大规模数据（n_list 很大时）

---

## 3. 距离度量

### 3.1 常用距离度量

#### 3.1.1 Euclidean Distance (L2)

$$
d_{L2}(x, y) = \|x - y\|_2 = \sqrt{\sum_{i=1}^{d} (x_i - y_i)^2}
$$

**性质**：
- 满足三角不等式：$d(x, z) \leq d(x, y) + d(y, z)$
- 对向量尺度敏感（受向量长度影响）
- 适合位置敏感的搜索

#### 3.1.2 Squared Euclidean Distance (L2²)

$$
d_{L2^2}(x, y) = \|x - y\|_2^2 = \sum_{i=1}^{d} (x_i - y_i)^2
$$

**性质**：
- 与 L2 排序相同（单调变换）
- 计算更快（省去 sqrt）
- FAISS 默认使用 L2²

#### 3.1.3 Cosine Similarity

$$
\cos(x, y) = \frac{x \cdot y}{\|x\| \cdot \|y\|} = \frac{\sum_{i=1}^{d} x_i y_i}{\sqrt{\sum x_i^2} \cdot \sqrt{\sum y_i^2}}
$$

**Cosine Distance**：
$$
d_{cos}(x, y) = 1 - \cos(x, y)
$$

**性质**：
- 仅关注方向，不受向量长度影响
- 范围：$[0, 2]$（相似时接近 0）
- 文本嵌入常用（normalized vectors）

#### 3.1.4 Inner Product (Dot Product)

$$
IP(x, y) = x \cdot y = \sum_{i=1}^{d} x_i y_i
$$

**Maximum Inner Product Search (MIPS)**：
- 寻找最大 inner product 的向量
- 用于推荐系统（用户-物品匹配）

**性质**：
- 对向量长度敏感
- 归一化后等价于 cosine：$IP(x/||x||, y/||y||) = \cos(x, y)$

### 3.2 度量转换关系

**FAISS 支持的转换表**：

| 源度量 | 目标度量 | 转换方法 |
|-------|---------|---------|
| L2 → IP | 添加维度：$x' = [x; 0], y' = [y; \sqrt{\alpha^2 - \|y\|^2}]$ | 查表法 |
| L2 → Cosine | 归一化：$x' = x/\|x\|, y' = y/\|y\|$ | 预处理 |
| IP → L2 | 添加维度：$x' = [x; -\alpha/2], y' = [y; \|y\|^2/\alpha]$ | 查表法 |
| IP → Cosine | 归一化 | 预处理 |
| Cosine → L2 | 归一化 + L2 | 预处理 |
| Cosine → IP | 归一化 + IP | 预处理 |

**关键洞察**：
- **归一化向量**：Cosine、IP、L2 排序相同
- **推荐系统用 IP**：未归一化的向量更合适
- **文本搜索用 Cosine**：大多数 embedding 模型训练时使用 cosine

### 3.3 距离计算优化技巧

#### 3.3.1 SIMD (Single Instruction Multiple Data)

利用 SIMD 指令同时计算多个距离：

```cpp
// AVX2 实现 L2 距离（同时计算 8 个维度）
float l2_distance_avx2(const float* x, const float* y, int d) {
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < d; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        __m256 diff = _mm256_sub_ps(vx, vy);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }
    // 横向求和
    float result[8];
    _mm256_storeu_ps(result, sum);
    return sqrt(result[0] + result[1] + ... + result[7]);
}
```

**加速效果**：8× 理论加速

#### 3.3.2 查表法 (Lookup Table)

用于 PQ 距离计算：

```cpp
// 预计算查询到码字的距离表
float LUT[m][k_star];  // m个子空间，每个k_star个码字

for (int j = 0; j < m; j++) {
    for (int i = 0; i < k_star; i++) {
        // u_j^q是查询的子向量，c[j][i]是码字
        LUT[j][i] = l2_distance(u_j^q, c[j][i], D/m);
    }
}

// 计算到编码向量的距离（仅查表求和）
float dist = 0;
for (int j = 0; j < m; j++) {
    dist += LUT[j][code[j]];  // code[j]是向量在第j个子空间的编码
}
```

#### 3.3.3 矩阵乘法分解

批量查询时，将距离计算转换为矩阵乘法：

$$
\|q - x_i\|^2 = \|q\|^2 + \|x_i\|^2 - 2 q \cdot x_i
$$

其中 $q \cdot x_i$ 可用矩阵乘法批量计算：

$$
Q \cdot X^T = [q_1, ..., q_B] \cdot [x_1, ..., x_N]^T
$$

**优势**：利用 BLAS 库优化矩阵乘法

---

## 4. Hub Formation 问题

### 4.1 问题定义

**Hubness 现象**：在高维空间中，少数向量（hubs）出现在大量其他向量的 k-NN 列表中。

**度量**：$k$-occurrence 分布 $N_k$ 的偏度

$$
S_{N_k} = \frac{\mathbb{E}[(N_k - \mu_{N_k})^3]}{\sigma_{N_k}^3}
$$

**现象**：
- $S_{N_k} > 0$：存在 hubs（少数向量频繁出现）
- $S_{N_k} = 0$：均匀分布（无 hub 现象）
- $S_{N_k} < 0$：anti-hubs（某些向量从不出现）

### 4.2 Hub Formation 原因

#### 4.2.1 距离集中 (Distance Concentration)

高维空间中，所有向量间的距离趋于相似：

$$
\frac{\text{Var}(d)}{\mathbb{E}[d]} \to 0 \text{ as } d \to \infty
$$

**后果**：
- 距离失去区分能力
- 少数"平均向量"距离所有向量都较近
- 这些向量成为 hubs

#### 4.2.2 几何直觉

在高维空间：
- 大部分数据点分布在空间的"外壳"（超球面附近）
- 靠近中心的点距离所有点都较近
- 这些中心点成为 hubs

**可视化**（简化版）：

```
高维空间分布：

    ←─ 超球面外壳 ─→
    
        *   *      ← 大多数点在外壳
      *       *
     *  [HUB] *    ← Hub 在中心附近
      *       *
        *   *
        
Hub 到所有点的距离都较短
→ Hub 频繁出现在 k-NN 列表中
```

### 4.3 Hub 对 ANN 搜索的影响

#### 4.3.1 负面影响

1. **Recall 降低**：
   - Hubs 占据大量 k-NN 位置
   - 真正相关的点被 hubs 替代

2. **搜索路径集中**：
   - 图搜索频繁经过 hub nodes
   - 搜索多样性降低

#### 4.3.2 正面影响（Hub Highway Hypothesis）

**论文发现**：arxiv 2412.01940 "Down with the Hierarchy"

**Hub Highway Hypothesis**：
- 在高维图索引中，hubs 形成天然"高速公路"
- 搜索早期快速经过 hubs 到达目标区域
- Hierarchy 在高维不再需要，因为 hubs 已提供快速跳转功能

**证据**：
- FlatNav（去掉 hierarchy 的 HNSW）与 HNSW 性能相当
- 高维数据中，查询早期 80% 时间经过 top 5% 的 hub nodes
- Hub nodes 连接度高，形成良好路由结构

### 4.4 Hubness Reduction 方法

#### 4.4.1 Local Scaling

对每个向量使用局部尺度：

$$
d_{LS}(x, y) = \frac{d(x, y)}{s_x \cdot s_y}
$$

其中 $s_x$ 是向量 $x$ 到其 k-NN 的平均距离。

#### 4.4.2 Mutual Proximity

计算相对距离：

$$
MP(x, y) = P(d(x, z) > d(x, y)) \cdot P(d(y, z) > d(x, y))
$$

#### 4.4.3 DisSim Local Scaling

结合距离和局部尺度：

$$
d_{DSL}(x, y) = d(x, y) - \mu_x - \mu_y
$$

---

## 5. 高维空间的距离集中现象

### 5.1 数学表述

**距离集中定理**：

设 $x, y \in \mathbb{R}^d$，各分量独立同分布（如均匀分布或正态分布），则：

$$
\mathbb{E}[d(x, y)] \sim \sqrt{d}
$$

$$
\text{Var}(d(x, y)) \to \text{const} \text{ as } d \to \infty
$$

**相对方差**：

$$
\frac{\sigma_d}{\mu_d} \sim \frac{1}{\sqrt{d}} \to 0
$$

**含义**：当维度增加，所有距离趋于相同的期望值。

### 5.2 对 ANN 搜索的影响

1. **索引效果降低**：
   - 树结构（kd-tree）在高维失效
   - 分支裁剪不再有效

2. **距离排序困难**：
   - 最近和最远的距离差异缩小
   - 难以区分"真正相似"的向量

3. **图索引仍有效**：
   - 图索引通过局部连接避免全局距离问题
   - 但需要处理 hub 现象

### 5.3 降维作为缓解方法

**PCA + 索引**：
- PCA 降低维度，保留主要方差方向
- 在低维空间索引效果更好

**Johnson-Lindenstrauss 投影**：
- 随机投影到低维，保持相对距离
- 保证距离关系的近似

---

## 6. 与流形几何的结合点

### 6.1 本征维度 (Intrinsic Dimensionality)

**概念**：
- 高维数据可能嵌入在低维流形上
- 本征维度 $d_{int} < d$（名义维度）

**应用**：
- 若 $d_{int}$ 较低，ANN 搜索更有效
- 可用流形学习估算 $d_{int}$

### 6.2 测地距离 vs Euclidean 距离

**Euclidean 距离的问题**：
- 不考虑流形结构
- 在弯曲流形上不准确

**测地距离**：
- 流形上的最短路径
- 更准确地反映语义相似性

**结合方案**：
1. Isomap 计算测地距离
2. 用测地距离构建 HNSW 图
3. 搜索时考虑流形结构

### 6.3 曲率与 Hub Formation

**推测关系**：
- 正曲率流形（如球面）：数据集中在边界，hub 可能位于中心
- 负曲率流形（如双曲空间）：边界扩张快，hub 分布不同

**双曲嵌入**：
- 双曲空间适合层级结构
- HNSW 的层级结构天然契合双曲几何
- 可在双曲空间构建索引

### 6.4 双曲空间索引构建

**Poincaré 球模型距离**：

$$
d_{\mathbb{D}}(x, y) = \text{arccosh}\left(1 + 2\frac{\|x - y\|^2}{(1 - \|x\|^2)(1 - \|y\|^2)}\right)
$$

**潜在方案**：
- 将层级数据嵌入双曲空间
- 在双曲空间构建 HNSW 图
- 利用双曲空间的特性优化连接

### 6.5 流形学习与索引预处理

**预处理流程**：

```
原始高维数据
    ↓
流形学习（Isomap/LLE/t-SNE）
    ↓
低维嵌入（保留流形结构）
    ↓
ANN 索引构建（HNSW/IVF）
    ↓
高效搜索
```

**优势**：
- 降低本征维度，提高索引效率
- 保留数据的几何结构
- 减少 hub 现象的影响

---

## 7. 伪代码实现

### 7.1 HNSW 插入伪代码

```python
def hnsw_insert(index, q, M, M_max0, ef_construction, m_L):
    """
    向 HNSW 索引插入新向量 q
    
    Args:
        index: HNSW 索引结构
        q: 新向量
        M: 每层最大连接数
        M_max0: 底层最大连接数
        ef_construction: 构建时搜索宽度
        m_L: 层级因子
    """
    # 1. 随机分配层级
    l = random_layer(m_L)  # P(l) = (1-m_L^{-1}) * m_L^{-l}
    
    # 2. 从最高层开始搜索入口点
    ep = index.entry_point  # 入口点
    L = index.max_layer
    
    # 3. 高层搜索：找到各层的入口点
    for lc in range(L, l + 1):  # 从最高层到 l 层
        W = search_layer(q, [ep], ef_construction, lc)
        ep = nearest_in(W)  # 更新入口点
    
    # 4. 从 l 层到底层：插入并连接
    for lc in range(l, -1):  # 从 l 层到 0 层
        W = search_layer(q, [ep], ef_construction, lc)
        
        # 选择邻居（启发式）
        neighbors = select_neighbors_heuristic(q, W, M if lc > 0 else M_max0)
        
        # 添加双向连接
        add_bidirectional_connections(index, q, neighbors, lc)
        
        # 简化邻居连接（防止连接过多）
        for neighbor in neighbors:
            simplify_connections(index, neighbor, M if lc > 0 else M_max0, lc)
        
        # 更新入口点（下一层）
        ep = nearest_in(W)
    
    # 5. 更新索引入口点（如果是新最高层）
    if l > L:
        index.entry_point = q
        index.max_layer = l
```

### 7.2 HNSW 搜索伪代码

```python
def hnsw_search(index, q, ef, K):
    """
    在 HNSW 索引中搜索 q 的 K 个最近邻居
    
    Args:
        index: HNSW 索引
        q: 查询向量
        ef: 搜索宽度
        K: 返回数量
    
    Returns:
        list of (id, distance) tuples
    """
    ep = index.entry_point
    L = index.max_layer
    
    # 高层快速定位
    for lc in range(L, 1):
        W = search_layer(q, [ep], ef=1, layer=lc)  # 高层用 ef=1
        ep = nearest_in(W)
    
    # 底层精确搜索
    W = search_layer(q, [ep], ef, layer=0)
    
    # 返回最近的 K 个
    return sorted(W, key=lambda x: x.distance)[:K]


def search_layer(q, entry_points, ef, layer):
    """
    在单层搜索
    
    Returns:
        list of candidates with distances
    """
    visited = set(entry_points)
    candidates = PriorityQueue(entry_points)  # 最小堆，按距离
    results = PriorityQueue(entry_points)     # 最大堆，按距离
    
    while candidates:
        c = candidates.pop_nearest()
        f = results.pop_farthest()
        
        # 提前终止条件
        if c.dist > f.dist and len(results) == ef:
            break
        
        # 遍历邻居
        for e in get_neighbors(c, layer):
            if e not in visited:
                visited.add(e)
                f = results.get_farthest()
                
                e_dist = distance(q, e.vector)
                if e_dist < f.dist or len(results) < ef:
                    candidates.add(e, e_dist)
                    results.add(e, e_dist)
                    
                    if len(results) > ef:
                        results.remove_farthest()
    
    return results.elements()
```

### 7.3 IVF-PQ 搜索伪代码

```python
def ivf_pq_search(index, q, n_probe, K):
    """
    IVF-PQ 索引搜索
    
    Args:
        index: IVF-PQ 索引
        q: 查询向量
        n_probe: 扫描 cell 数
        K: 返回数量
    """
    # 1. 找最近的 n_probe 个 cell
    cell_distances = [distance(q, c.centroid) for c in index.cells]
    nearest_cells = sorted_cells_by_distance(cell_distances)[:n_probe]
    
    # 2. 预计算 PQ 查表
    LUT = build_lut(q, index.pq_codebooks)
    
    # 3. 在 cell 内搜索
    candidates = []
    for cell in nearest_cells:
        for vec_id, pq_code in cell.vectors:
            dist = sum(LUT[j][pq_code[j]] for j in range(index.m))
            candidates.append((vec_id, dist))
    
    # 4. 可选：重排序（精确计算）
    if index.reorder:
        candidates = rerank(q, candidates[:K * 10], K)
    
    return sorted(candidates)[:K]


def build_lut(q, codebooks):
    """
    构建 PQ 查表
    
    Args:
        q: 查询向量
        codebooks: PQ 码本 [m][k_star][D/m]
    
    Returns:
        LUT: [m][k_star] 查表
    """
    m = len(codebooks)
    D_star = len(codebooks[0][0])  # D/m
    k_star = len(codebooks[0])
    
    LUT = [[0] * k_star for _ in range(m)]
    
    for j in range(m):
        u_q = q[j * D_star : (j+1) * D_star]  # 子向量
        for i in range(k_star):
            LUT[j][i] = l2_distance(u_q, codebooks[j][i])
    
    return LUT
```

### 7.4 距离计算优化伪代码

```python
def batch_l2_distance(Q, X):
    """
    批量计算 L2 距离（利用矩阵乘法）
    
    Args:
        Q: 查询矩阵 [B, D]
        X: 数据矩阵 [N, D]
    
    Returns:
        distances: [B, N]
    """
    # 分解：||q - x||² = ||q||² + ||x||² - 2*q·x
    
    # 1. 查询范数
    Q_norm = np.sum(Q ** 2, axis=1, keepdims=True)  # [B, 1]
    
    # 2. 数据范数
    X_norm = np.sum(X ** 2, axis=1, keepdims=True)  # [N, 1]
    
    # 3. 内积（矩阵乘法）
    Q_dot_X = Q @ X.T  # [B, N] - 利用 BLAS
    
    # 4. 组合
    distances = Q_norm + X_norm.T - 2 * Q_dot_X
    
    return np.sqrt(np.maximum(distances, 0))  # 防止负数
```

---

## 8. 参考文献

### HNSW 相关

1. **Malkov & Yashunin (2016)**: "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" - HNSW 原论文
2. **Malkov et al. (2020)**: "HNSWlib" - 开源实现
3. **BlaiseMuhirwa et al. (2024)**: "Down with the Hierarchy: The 'H' in HNSW Stands for 'Hubs'" - arxiv 2412.01940

### FAISS 相关

4. **Jégou et al. (2010)**: "Product Quantization for Nearest Neighbor Search"
5. **Johnson et al. (2017)**: "Billion-scale similarity search with GPUs"
6. **Douze et al. (2024)**: "The Faiss Library" - arxiv 2401.08281

### Hubness 相关

7. **Radovanovic et al. (2010)**: "Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data"
8. **François et al. (2007)**: "The concentration of fractional distances"

### 距离度量相关

9. **Aggarwal et al. (2001)**: "On the surprising behavior of distance metrics in high dimensional spaces"
10. **Pinecone Learn**: "Product Quantization: Compressing high-dimensional vectors by 97%"

### 流形几何相关

11. **Tenenbaum et al. (2000)**: "A Global Geometric Framework for Nonlinear Dimensionality Reduction" - Isomap
12. **Nickel & Kiela (2017)**: "Poincaré Embeddings for Learning Hierarchical Representations"

---

## 附录：核心公式速查表

| 公式 | 表达式 |
|-----|--------|
| **HNSW 层级分布** | $P(l) = (1-m_L^{-1}) \cdot m_L^{-l}$ |
| **Euclidean Distance** | $d_{L2}(x,y) = \sqrt{\sum_i (x_i - y_i)^2}$ |
| **Cosine Distance** | $d_{cos}(x,y) = 1 - \frac{x \cdot y}{\|x\| \|y\|}$ |
| **PQ 编码** | $q(x) = [\arg\min_{c \in C_1} \|u_1 - c\|, ..., \arg\min_{c \in C_m} \|u_m - c\|]$ |
| **ADC 距离** | $\|q - x\|^2 \approx \sum_{j=1}^{m} \|u_j^q - c_{j,q_j(x)}\|^2$ |
| **Hubness 偏度** | $S_{N_k} = \mathbb{E}[(N_k - \mu)^3] / \sigma^3$ |
| **距离集中** | $\sigma_d / \mu_d \sim 1/\sqrt{d} \to 0$ |
| **Poincaré 距离** | $d_{\mathbb{D}}(x,y) = \text{arccosh}(1 + 2\frac{\|x-y\|^2}{(1-\|x\|^2)(1-\|y\|^2)})$ |

---

*文档完成于 2026-04-20*
*ClawTeam manifold-vector 研究组*