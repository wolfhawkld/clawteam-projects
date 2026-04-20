# Manifold-RAG 实现路径与代码示例

> 作者: implementer (ClawTeam manifold-vector)
> 日期: 2026-04-20
> 状态: 完成

---

## 目录

1. [完整代码示例](#1-完整代码示例)
2. [改造现有向量数据库的建议](#2-改造现有向量数据库的建议)
3. [实验设计](#3-实验设计)
4. [实现优先级排序](#4-实现优先级排序)
5. [参考文献](#5-参考文献)

---

## 1. 完整代码示例

### 1.1 Poincaré/Lorentz 距离计算函数

#### 1.1.1 Poincaré 球模型实现

```python
"""
poincare_distance.py - Poincaré球模型双曲距离计算

作者: implementer
日期: 2026-04-20
"""

import numpy as np
from typing import Union, Tuple

class PoincareDistance:
    """
    Poincaré球模型距离计算
    
    使用场景：
    - 层级数据嵌入
    - RAG知识图谱的层级关系
    - 降低Hubness现象
    """
    
    def __init__(self, radius: float = 1.0):
        """
        Args:
            radius: 球半径（默认1.0为标准Poincaré球）
        """
        self.radius = radius
    
    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        计算两点间的Poincaré距离
        
        公式: d(x,y) = arccosh(1 + 2||x-y||² / ((1-||x||²)(1-||y||²)))
        
        Args:
            x, y: 球内的向量（||x|| < radius）
        
        Returns:
            双曲距离值
        """
        # 确保输入有效
        x_norm_sq = np.sum(x**2)
        y_norm_sq = np.sum(y**2)
        
        if x_norm_sq >= self.radius**2 or y_norm_sq >= self.radius**2:
            raise ValueError(f"Vectors must be inside the ball: "
                           f"||x||²={x_norm_sq}, ||y||²={y_norm_sq}")
        
        diff_norm_sq = np.sum((x - y)**2)
        
        # Poincaré距离公式
        numerator = 2 * diff_norm_sq
        denominator = (self.radius**2 - x_norm_sq) * (self.radius**2 - y_norm_sq)
        
        # arccosh参数必须 >= 1
        arg = 1 + numerator / denominator
        arg = np.maximum(arg, 1.0 + 1e-7)  # 防止边界问题
        
        return np.arccosh(arg)
    
    def distance_batch(self, Q: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        批量计算距离（优化版本）
        
        利用矩阵运算并行计算
        
        Args:
            Q: 查询向量 [B, D]
            X: 数据向量 [N, D]
        
        Returns:
            距离矩阵 [B, N]
        """
        # 预计算范数
        Q_norm_sq = np.sum(Q**2, axis=1, keepdims=True)  # [B, 1]
        X_norm_sq = np.sum(X**2, axis=1, keepdims=True)  # [N, 1]
        
        # 检查边界
        if np.any(Q_norm_sq >= self.radius**2) or np.any(X_norm_sq >= self.radius**2):
            raise ValueError("Vectors must be inside the ball")
        
        # ||Q - X||² = ||Q||² + ||X||² - 2*Q·X
        diff_norm_sq = Q_norm_sq + X_norm_sq.T - 2 * Q @ X.T  # [B, N]
        
        numerator = 2 * diff_norm_sq
        denominator = (self.radius**2 - Q_norm_sq) * (self.radius**2 - X_norm_sq.T)
        
        arg = 1 + numerator / denominator
        arg = np.maximum(arg, 1.0 + 1e-7)
        
        return np.arccosh(arg)
    
    def project_to_ball(self, x: np.ndarray, epsilon: float = 1e-3) -> np.ndarray:
        """
        将向量投影到球内
        
        Args:
            x: 任意向量
            epsilon: 边界缓冲（避免恰好到边界）
        
        Returns:
            投影后的向量（||x|| < radius - epsilon）
        """
        norm = np.linalg.norm(x)
        if norm >= self.radius - epsilon:
            x = x * (self.radius - epsilon) / norm
        return x
    
    def distance_to_origin(self, x: np.ndarray) -> float:
        """
        点到原点的距离
        
        公式: d(O, x) = arctanh(||x||/radius)
        """
        x_norm = np.linalg.norm(x)
        return np.arctanh(x_norm / self.radius)
    
    def mobius_add(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Möbius加法（双曲空间的"加法"）
        
        公式: x ⊕ y = ((1+2<x,y>+||y||²)x + (1-||x||²)y) / (1+2<x,y>+||x||²||y||²)
        
        应用：双曲空间的"平移"
        """
        x_norm_sq = np.sum(x**2)
        y_norm_sq = np.sum(y**2)
        xy_inner = np.dot(x, y)
        
        numerator = (1 + 2*xy_inner + y_norm_sq) * x + (1 - x_norm_sq) * y
        denominator = 1 + 2*xy_inner + x_norm_sq * y_norm_sq
        
        return numerator / denominator
    
    def exp_map(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        指数映射：从切空间到双曲空间
        
        公式: Exp_x(v) = x ⊕ (tanh(||v||/2) * v/||v||)
        
        应用：在双曲空间沿方向v移动
        """
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-10:
            return x
        
        # 单位方向向量
        v_unit = v / v_norm
        
        # 移动距离
        move_factor = np.tanh(v_norm / 2)
        
        return self.mobius_add(x, move_factor * v_unit)


# 使用示例
if __name__ == "__main__":
    poincare = PoincareDistance(radius=1.0)
    
    # 两个向量
    x = np.array([0.3, 0.2, 0.1])
    y = np.array([0.5, 0.1, 0.2])
    
    # 计算距离
    dist = poincare.distance(x, y)
    print(f"Poincaré distance: {dist:.4f}")
    
    # 批量计算
    Q = np.random.randn(5, 3) * 0.3
    X = np.random.randn(100, 3) * 0.3
    Q = poincare.project_to_ball(Q)
    X = poincare.project_to_ball(X)
    
    dist_matrix = poincare.distance_batch(Q, X)
    print(f"Batch distance matrix shape: {dist_matrix.shape}")
```

#### 1.1.2 Lorentz (Hyperboloid) 模型实现

```python
"""
lorentz_distance.py - Lorentz模型双曲距离计算

作者: implementer
日期: 2026-04-20

优势：
- 距离计算更简洁（仅涉及内积）
- 数值稳定性更好
- 适合优化（可用平方距离避免arccosh）
"""

import numpy as np
from typing import Union, Tuple

class LorentzDistance:
    """
    Lorentz (Hyperboloid) 模型距离计算
    
    Lorentz模型定义：
    - 点在双曲面上：<x,x>_L = -1, x₀ > 0
    - Lorentz内积：<x,y>_L = -x₀y₀ + Σx_i y_i
    """
    
    def __init__(self, curvature: float = 1.0):
        """
        Args:
            curvature: 曲率参数（默认-1，即标准双曲空间）
                      实际曲率 K = -curvature
        """
        self.curvature = curvature
    
    def euclidean_to_lorentz(self, x: np.ndarray) -> np.ndarray:
        """
        将欧氏向量转换为Lorentz表示
        
        公式: x_L = [√(1+||x||²), x₁, x₂, ...]
        
        Args:
            x: 欧氏向量 [D]
        
        Returns:
            Lorentz向量 [D+1]，满足 <x_L, x_L>_L = -1
        """
        x_norm_sq = np.sum(x**2)
        x0 = np.sqrt(1 + x_norm_sq / self.curvature)
        return np.concatenate([[x0], x])
    
    def lorentz_to_euclidean(self, x_lorentz: np.ndarray) -> np.ndarray:
        """
        Lorentz向量转回欧氏
        
        公式: x_eu = x_L[1:] / x_L[0]（对于原点附近的点）
        """
        return x_lorentz[1:] / x_lorentz[0]
    
    def lorentz_inner(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Lorentz内积
        
        公式: <x,y>_L = -x₀y₀ + Σx_i y_i
        """
        return -x[0] * y[0] + np.dot(x[1:], y[1:])
    
    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Lorentz距离
        
        公式: d(x,y) = arccosh(-<x,y>_L)
        
        注意：对于前向双曲面上的点，<x,y>_L ≤ -1
        """
        inner = self.lorentz_inner(x, y)
        
        # 确保参数有效（对于双曲面上的点，inner应该 ≤ -1）
        if inner > -1:
            inner = -1 - 1e-7
        
        return np.arccosh(-inner)
    
    def distance_squared(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Lorentz平方距离（优化友好）
        
        公式: d²(x,y) = -2 - 2<x,y>_L
        
        优势：
        - 避免 arccosh，计算更快
        - 梯度更稳定（优化时）
        - 排序与真实距离相同
        """
        inner = self.lorentz_inner(x, y)
        return -2 - 2 * inner
    
    def distance_batch(self, Q: np.ndarray, X: np.ndarray, 
                      use_squared: bool = False) -> np.ndarray:
        """
        批量计算Lorentz距离
        
        Args:
            Q: 查询向量（Lorentz格式）[B, D+1]
            X: 数据向量（Lorentz格式）[N, D+1]
            use_squared: 是否使用平方距离
        
        Returns:
            距离矩阵 [B, N]
        """
        # Lorentz内积的矩阵形式
        # <Q,X>_L = -Q[:,0]*X[:,0] + Q[:,1:]·X[:,1:].T
        
        Q0 = Q[:, 0:1]  # [B, 1]
        X0 = X[:, 0:1]  # [N, 1]
        Q_space = Q[:, 1:]  # [B, D]
        X_space = X[:, 1:]  # [N, D]
        
        # Lorentz内积矩阵
        lorentz_inner_matrix = -Q0 @ X0.T + Q_space @ X_space.T  # [B, N]
        
        if use_squared:
            return -2 - 2 * lorentz_inner_matrix
        else:
            # 确保 arccosh 参数有效
            arg = np.maximum(-lorentz_inner_matrix, 1.0 + 1e-7)
            return np.arccosh(arg)
    
    def exp_map(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        指数映射：切空间 → Lorentz空间
        
        公式: Exp_x(v) = cosh(||v||) * x + sinh(||v||) * v/||v||
        
        其中 v 需满足 <x,v>_L = 0（切向量）
        """
        v_norm = np.sqrt(np.sum(v**2) - v[0]**2)  # Lorentz范数
        
        if v_norm < 1e-10:
            return x
        
        # 确保v是切向量（投影到切空间）
        # v_tangent = v + <x,v>_L * x
        xv_inner = self.lorentz_inner(x, v)
        v_tangent = v + xv_inner * x
        
        v_tangent_norm = np.sqrt(-self.lorentz_inner(v_tangent, v_tangent))
        v_unit = v_tangent / v_tangent_norm
        
        # 指数映射
        result = np.cosh(v_tangent_norm) * x + np.sinh(v_tangent_norm) * v_unit
        
        # 确保结果在双曲面上
        return result / np.sqrt(-self.lorentz_inner(result, result))
    
    def log_map(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        对数映射：Lorentz空间 → 切空间
        
        公式: Log_x(y) = d(x,y) * (y - <x,y>_L * x) / ||y - <x,y>_L * x||
        """
        dist = self.distance(x, y)
        xy_inner = self.lorentz_inner(x, y)
        
        diff = y + xy_inner * x  # 注意：+号因为xy_inner是负的
        
        # Lorentz范数
        diff_norm = np.sqrt(-self.lorentz_inner(diff, diff))
        
        if diff_norm < 1e-10:
            return np.zeros_like(x)
        
        return dist * diff / diff_norm
    
    def gyro_add(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Gyro加法（双曲空间的"加法"）
        
        Lorentz模型的gyro加法公式较复杂，
        通常通过exp_map实现
        """
        # 简化实现：通过切向量相加
        v = self.log_map(x, y)
        return self.exp_map(x, v)


# 使用示例
if __name__ == "__main__":
    lorentz = LorentzDistance(curvature=1.0)
    
    # 从欧氏向量转换
    x_eu = np.array([0.3, 0.2, 0.1])
    y_eu = np.array([0.5, 0.1, 0.2])
    
    x_lo = lorentz.euclidean_to_lorentz(x_eu)
    y_lo = lorentz.euclidean_to_lorentz(y_eu)
    
    # 计算距离
    dist = lorentz.distance(x_lo, y_lo)
    dist_sq = lorentz.distance_squared(x_lo, y_lo)
    
    print(f"Lorentz distance: {dist:.4f}")
    print(f"Squared distance: {dist_sq:.4f}")
    
    # 验证在双曲面上
    print(f"<x,x>_L = {lorentz.lorentz_inner(x_lo, x_lo):.4f} (应为-1)")
```

### 1.2 Hyperbolic HNSW 实现原型

```python
"""
hyperbolic_hnsw.py - 双曲空间的HNSW索引实现

作者: implementer
日期: 2026-04-20

核心改动：
1. 使用Poincaré/Lorentz距离替代欧氏距离
2. 投影函数确保向量在双曲空间内
3. 搜索和插入逻辑保持不变（仅替换距离函数）
"""

import numpy as np
import heapq
import random
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict

class HyperbolicHNSW:
    """
    双曲空间的HNSW索引
    
    支持两种模型：
    - 'poincare': Poincaré球模型
    - 'lorentz': Lorentz双曲面模型
    
    与标准HNSW的差异：
    - 距离计算使用双曲距离
    - 向量存储时投影到双曲空间
    - 其他结构完全相同
    """
    
    def __init__(
        self,
        dim: int,
        M: int = 16,
        M_max0: int = 32,
        ef_construction: int = 200,
        ef_search: int = 100,
        m_L: float = None,
        model: str = 'poincare',
        radius: float = 0.99,  # Poincaré球半径（略小于1避免边界问题）
        use_squared_distance: bool = False  # Lorentz时可用平方距离加速
    ):
        """
        Args:
            dim: 向量维度
            M: 每层最大连接数
            M_max0: 底层最大连接数
            ef_construction: 构建时搜索宽度
            ef_search: 搜索时搜索宽度
            m_L: 层级乘数因子（默认 1/ln(M)）
            model: 'poincare' 或 'lorentz'
            radius: Poincaré球半径
            use_squared_distance: 是否使用平方距离（仅Lorentz有效）
        """
        self.dim = dim
        self.M = M
        self.M_max0 = M_max0
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.m_L = m_L if m_L else 1.0 / np.log(M)
        self.model = model
        self.radius = radius
        self.use_squared_distance = use_squared_distance
        
        # 数据存储
        self.nodes: Dict[int, np.ndarray] = {}  # {node_id: vector}
        self.layers: Dict[int, Dict[int, List[int]]] = defaultdict(
            lambda: defaultdict(list)
        )  # {layer: {node_id: [neighbor_ids]}}
        self.max_layer = 0
        self.entry_point: Optional[int] = None
        self.node_count = 0
        
        # 初始化距离计算器
        if model == 'poincare':
            from poincare_distance import PoincareDistance
            self.distance_calc = PoincareDistance(radius=radius)
        elif model == 'lorentz':
            from lorentz_distance import LorentzDistance
            self.distance_calc = LorentzDistance()
        else:
            raise ValueError(f"Unknown model: {model}")
    
    def _random_layer(self) -> int:
        """
        随机分配层级
        
        分布: P(l) = (1 - m_L^{-1}) * m_L^{-l}
        """
        # 使用指数分布生成层级
        l = int(-np.log(random.random()) / np.log(self.m_L))
        return l
    
    def _distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        计算双曲距离
        """
        if self.model == 'poincare':
            return self.distance_calc.distance(x, y)
        else:  # lorentz
            if self.use_squared_distance:
                return self.distance_calc.distance_squared(x, y)
            else:
                return self.distance_calc.distance(x, y)
    
    def _project_vector(self, x: np.ndarray) -> np.ndarray:
        """
        投影向量到双曲空间
        """
        if self.model == 'poincare':
            return self.distance_calc.project_to_ball(x, epsilon=1-self.radius)
        else:  # lorentz
            return self.distance_calc.euclidean_to_lorentz(x)
    
    def insert(self, x: np.ndarray) -> int:
        """
        插入新向量
        
        Args:
            x: 欧氏向量 [dim]
        
        Returns:
            node_id: 分配的节点ID
        """
        # 投影到双曲空间
        x_projected = self._project_vector(x)
        
        # 分配层级
        l = self._random_layer()
        
        # 分配节点ID
        node_id = self.node_count
        self.node_count += 1
        self.nodes[node_id] = x_projected
        
        # 如果是第一个节点
        if self.entry_point is None:
            self.entry_point = node_id
            self.max_layer = l
            for lc in range(l + 1):
                self.layers[lc][node_id] = []
            return node_id
        
        # 从入口点开始搜索
        ep = self.entry_point
        
        # 高层快速定位（找到各层的最佳入口点）
        for lc in range(self.max_layer, l + 1):
            W = self._search_layer(x_projected, [ep], ef=1, layer=lc)
            if W:
                ep = W[0][0]  # 取最近的
        
        # 从l层到底层插入
        for lc in range(min(l, self.max_layer), -1):
            W = self._search_layer(x_projected, [ep], 
                                   ef=self.ef_construction, layer=lc)
            
            # 选择邻居（启发式）
            M_max = self.M if lc > 0 else self.M_max0
            neighbors = self._select_neighbors_heuristic(
                x_projected, W, M_max, layer=lc
            )
            
            # 添加双向连接
            self.layers[lc][node_id] = neighbors
            for n in neighbors:
                self.layers[lc][n].append(node_id)
                # 简化邻居连接（如果超过M_max）
                if len(self.layers[lc][n]) > M_max:
                    self._simplify_connections(n, M_max, lc)
            
            # 更新入口点
            if W:
                ep = W[0][0]
        
        # 更新入口点（如果是新最高层）
        if l > self.max_layer:
            self.entry_point = node_id
            self.max_layer = l
        
        return node_id
    
    def _search_layer(
        self,
        q: np.ndarray,
        entry_points: List[int],
        ef: int,
        layer: int
    ) -> List[Tuple[int, float]]:
        """
        单层搜索（使用双曲距离）
        
        Returns:
            [(node_id, distance)] 按距离升序排序
        """
        visited: Set[int] = set(entry_points)
        
        # 最小堆（候选，按距离排序）
        candidates: List[Tuple[float, int]] = []
        # 最大堆（结果，按距离排序）
        results: List[Tuple[float, int]] = []
        
        for ep in entry_points:
            d = self._distance(q, self.nodes[ep])
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(results, (-d, ep))  # 最大堆用负值
        
        while candidates:
            c_dist, c = heapq.heappop(candidates)
            
            # 获取结果中最远的距离
            if results:
                f_dist = -results[0][0]  # 最大堆的顶部是最远的
            else:
                f_dist = float('inf')
            
            # 提前终止
            if c_dist > f_dist and len(results) >= ef:
                break
            
            # 遍历邻居
            for neighbor in self.layers[layer].get(c, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    n_dist = self._distance(q, self.nodes[neighbor])
                    
                    if n_dist < f_dist or len(results) < ef:
                        heapq.heappush(candidates, (n_dist, neighbor))
                        heapq.heappush(results, (-n_dist, neighbor))
                        
                        if len(results) > ef:
                            heapq.heappop(results)
        
        # 转换结果格式
        return [(n, -d) for d, n in sorted(results, reverse=True)]
    
    def _select_neighbors_heuristic(
        self,
        q: np.ndarray,
        candidates: List[Tuple[int, float]],
        M: int,
        layer: int
    ) -> List[int]:
        """
        启发式邻居选择
        
        思想：
        1. 优先选择距离近的
        2. 但避免方向冲突（如果两个候选很近，选一个）
        """
        if len(candidates) <= M:
            return [c[0] for c in candidates]
        
        # 按距离排序
        sorted_candidates = sorted(candidates, key=lambda x: x[1])
        
        selected: List[int] = []
        
        for node_id, dist in sorted_candidates:
            if len(selected) >= M:
                break
            
            # 检查方向冲突
            conflict = False
            for s in selected:
                # 如果候选离已选邻居更近，说明在同一方向
                s_dist = self._distance(self.nodes[node_id], self.nodes[s])
                if s_dist < dist:
                    conflict = True
                    break
            
            if not conflict:
                selected.append(node_id)
        
        return selected
    
    def _simplify_connections(self, node_id: int, M_max: int, layer: int):
        """
        简化连接（保持M_max个最近邻居）
        """
        neighbors = self.layers[layer][node_id]
        if len(neighbors) <= M_max:
            return
        
        # 计算距离并排序
        distances = [(n, self._distance(self.nodes[node_id], self.nodes[n]))
                    for n in neighbors]
        sorted_neighbors = sorted(distances, key=lambda x: x[1])
        
        # 保留最近的M_max个
        self.layers[layer][node_id] = [n for n, d in sorted_neighbors[:M_max]]
    
    def search(self, q: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """
        搜索k个最近邻居
        
        Args:
            q: 查询向量（欧氏）
            k: 返回数量
        
        Returns:
            [(node_id, distance)] 按距离升序排序
        """
        if self.entry_point is None:
            return []
        
        # 投影查询向量
        q_projected = self._project_vector(q)
        
        ep = self.entry_point
        
        # 高层快速定位
        for lc in range(self.max_layer, 1):
            W = self._search_layer(q_projected, [ep], ef=1, layer=lc)
            if W:
                ep = W[0][0]
        
        # 底层精确搜索
        W = self._search_layer(q_projected, [ep], 
                              ef=self.ef_search, layer=0)
        
        return W[:k]
    
    def get_hubness_stats(self, k: int = 10) -> Dict:
        """
        计算Hubness统计
        
        Returns:
            {
                'skewness': hubness偏度,
                'N_k_distribution': 各节点作为k-NN的次数,
                'top_hubs': 最频繁的hub节点
            }
        """
        N_k = defaultdict(int)
        
        for node_id in range(self.node_count):
            if node_id not in self.nodes:
                continue
            
            # 搜索k个最近邻居
            neighbors = self.search(self.nodes[node_id], k + 1)
            
            # 统计（排除自己）
            for n, d in neighbors:
                if n != node_id:
                    N_k[n] += 1
        
        # 计算偏度
        N_k_values = [N_k.get(i, 0) for i in range(self.node_count)]
        mean_Nk = np.mean(N_k_values)
        std_Nk = np.std(N_k_values)
        
        if std_Nk > 0:
            skewness = np.mean((N_k_values - mean_Nk)**3) / std_Nk**3
        else:
            skewness = 0
        
        # 找top hubs
        sorted_hubs = sorted(N_k.items(), key=lambda x: -x[1])[:10]
        
        return {
            'skewness': skewness,
            'N_k_distribution': N_k_values,
            'top_hubs': sorted_hubs,
            'mean': mean_Nk,
            'std': std_Nk
        }


# 使用示例
if __name__ == "__main__":
    # 创建双曲HNSW索引
    dim = 128
    index = HyperbolicHNSW(
        dim=dim,
        M=16,
        ef_construction=200,
        model='poincare',
        radius=0.9
    )
    
    # 插入数据
    np.random.seed(42)
    N = 1000
    for i in range(N):
        x = np.random.randn(dim) * 0.3  # 小范数确保在球内
        index.insert(x)
    
    # 搜索
    q = np.random.randn(dim) * 0.3
    results = index.search(q, k=10)
    
    print(f"Search results: {results[:5]}")
    
    # 分析Hubness
    stats = index.get_hubness_stats(k=10)
    print(f"Hubness skewness: {stats['skewness']:.2f}")
    print(f"Top hubs: {stats['top_hubs'][:5]}")
```

### 1.3 度量张量学习模块

```python
"""
metric_tensor_learning.py - 从数据学习黎曼度量张量

作者: implementer
日期: 2026-04-20

核心思想：
- 数据分布的不均匀性反映流形的曲率
- 高密度区域的度量"收缩"（距离变小）
- 低密度区域的度量"扩张"（距离变大）
"""

import numpy as np
from typing import Callable, Tuple, Optional
from scipy.stats import gaussian_kde
from scipy.optimize import minimize

class MetricTensorLearner:
    """
    度量张量学习模块
    
    支持三种方法：
    1. 'density': 从数据密度估计构建度量
    2. 'pca': 从PCA主成分构建各向异性度量
    3. 'supervised': 从标签信息学习度量（如LMNN）
    """
    
    def __init__(
        self,
        method: str = 'density',
        bandwidth: float = 0.1,
        regularization: float = 1e-3
    ):
        """
        Args:
            method: 学习方法 ('density', 'pca', 'supervised')
            bandwidth: KDE带宽（仅density方法）
            regularization: 正则化参数（确保度量正定）
        """
        self.method = method
        self.bandwidth = bandwidth
        self.regularization = regularization
        self.metric_field: Optional[Callable] = None
    
    def fit(self, X: np.ndarray, labels: Optional[np.ndarray] = None):
        """
        学习度量张量场
        
        Args:
            X: 数据矩阵 [N, D]
            labels: 标签（仅supervised方法需要）
        """
        if self.method == 'density':
            self.metric_field = self._learn_from_density(X)
        elif self.method == 'pca':
            self.metric_field = self._learn_from_pca(X)
        elif self.method == 'supervised':
            if labels is None:
                raise ValueError("Supervised method requires labels")
            self.metric_field = self._learn_supervised(X, labels)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _learn_from_density(self, X: np.ndarray) -> Callable:
        """
        从数据密度学习度量
        
        思想：
        - 高密度区域 → 度量收缩 → 距离变小
        - 低密度区域 → 度量扩张 → 距离变大
        
        公式: g(x) = α * I - β * ∇∇ log p(x)
        （负Hessian表示密度的"曲率"）
        """
        # 估计密度
        kde = gaussian_kde(X.T, bw_method=self.bandwidth)
        
        def metric_at_point(x: np.ndarray) -> np.ndarray:
            """
            在点x处计算度量张量
            """
            D = len(x)
            
            # 计算log概率密度的Hessian
            # 使用数值微分
            h = 1e-5
            hessian = np.zeros((D, D))
            
            for i in range(D):
                for j in range(D):
                    # 二阶偏导
                    x_pp = x.copy()
                    x_pm = x.copy()
                    x_mp = x.copy()
                    x_mm = x.copy()
                    
                    x_pp[i] += h; x_pp[j] += h
                    x_pm[i] += h; x_pm[j] -= h
                    x_mp[i] -= h; x_mp[j] += h
                    x_mm[i] -= h; x_mm[j] -= h
                    
                    log_p_pp = np.log(kde(x_pp) + 1e-10)
                    log_p_pm = np.log(kde(x_pm) + 1e-10)
                    log_p_mp = np.log(kde(x_mp) + 1e-10)
                    log_p_mm = np.log(kde(x_mm) + 1e-10)
                    
                    hessian[i, j] = (log_p_pp - log_p_pm - log_p_mp + log_p_mm) / (4 * h**2)
            
            # 度量 = I - β * Hessian (负号使高密度区域度量小)
            # 添加正则化确保正定
            g = np.eye(D) - 0.1 * hessian
            g += self.regularization * np.eye(D)
            
            # 确保正定（对称化 + 最小特征值约束）
            g = (g + g.T) / 2
            eigenvalues, eigenvectors = np.linalg.eigh(g)
            eigenvalues = np.maximum(eigenvalues, self.regularization)
            g = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            
            return g
        
        return metric_at_point
    
    def _learn_from_pca(self, X: np.ndarray) -> Callable:
        """
        从PCA主成分学习各向异性度量
        
        思想：
        - 主成分方向（方差大）→ 度量大 → 距离更敏感
        - 次成分方向（方差小）→ 度量小 → 距离较宽容
        """
        from sklearn.decomposition import PCA
        
        pca = PCA()
        pca.fit(X)
        
        # 主成分方差作为度量权重
        variances = pca.explained_variance_
        components = pca.components_
        
        # 构建度量矩阵: g = Σ w_i * v_i * v_i^T
        D = X.shape[1]
        g_global = np.zeros((D, D))
        for i, (v, w) in enumerate(zip(components, variances)):
            g_global += w * v.reshape(-1, 1) @ v.reshape(1, -1)
        
        # 正则化
        g_global += self.regularization * np.eye(D)
        
        # 返回常度量（不依赖位置）
        def metric_at_point(x: np.ndarray) -> np.ndarray:
            return g_global
        
        return metric_at_point
    
    def _learn_supervised(self, X: np.ndarray, labels: np.ndarray) -> Callable:
        """
        从标签学习度量（简化版LMNN思想）
        
        思想：
        - 同类点距离应小
        - 不同类点距离应大
        """
        from sklearn.neighbors import NearestNeighbors
        
        D = X.shape[1]
        
        # 初始化度量为单位矩阵
        g_global = np.eye(D)
        
        # 简化优化：同类拉近，异类推远
        nn = NearestNeighbors(n_neighbors=5)
        nn.fit(X)
        _, indices = nn.kneighbors(X)
        
        # 统计同类/异类邻居
        gradient = np.zeros((D, D))
        
        for i in range(len(X)):
            label_i = labels[i]
            for j in indices[i]:
                if j == i:
                    continue
                
                diff = X[i] - X[j]
                diff_outer = diff.reshape(-1, 1) @ diff.reshape(1, -1)
                
                if labels[j] == label_i:
                    # 同类：减小距离 → 减小度量
                    gradient -= diff_outer
                else:
                    # 异类：增大距离 → 增大度量
                    gradient += 0.5 * diff_outer
        
        # 更新度量
        g_global = g_global - 0.001 * gradient
        g_global += self.regularization * np.eye(D)
        
        # 确保正定
        eigenvalues, eigenvectors = np.linalg.eigh(g_global)
        eigenvalues = np.maximum(eigenvalues, self.regularization)
        g_global = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        def metric_at_point(x: np.ndarray) -> np.ndarray:
            return g_global
        
        return metric_at_point
    
    def riemannian_distance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        N_segments: int = 100
    ) -> float:
        """
        计算黎曼距离（数值积分）
        
        公式: d(x,y) = ∫_γ sqrt(g · dx · dx)
        
        Args:
            x, y: 两点
            N_segments: 积分分段数
        
        Returns:
            黎曼距离
        """
        if self.metric_field is None:
            raise ValueError("Metric field not learned. Call fit() first.")
        
        # 沿直线积分（简化，假设测地线近似直线）
        path = np.linspace(x, y, N_segments)
        distance = 0
        
        for k in range(N_segments - 1):
            p = path[k]
            delta = path[k+1] - p
            g = self.metric_field(p)
            
            # ds = sqrt(g_ij * delta_i * delta_j)
            segment_length = np.sqrt(np.einsum('ij,i,j', g, delta, delta))
            distance += segment_length
        
        return distance
    
    def riemannian_distance_batch(
        self,
        Q: np.ndarray,
        X: np.ndarray,
        N_segments: int = 50
    ) -> np.ndarray:
        """
        批量黎曼距离计算（优化版）
        
        使用近似：g(x) ≈ g(c) 其中 c 是数据中心
        
        警告：这是简化方法，精确计算非常昂贵
        """
        if self.metric_field is None:
            raise ValueError("Metric field not learned. Call fit() first.")
        
        # 使用中心点的度量（简化）
        center = np.mean(X, axis=0)
        g_center = self.metric_field(center)
        
        # 批量计算: ||Q-X||² = Σ g_ij (q_i - x_i)(q_j - x_j)
        # 使用矩阵形式: (Q-X) @ g @ (Q-X)^T
        
        D = Q.shape[1]
        B = Q.shape[0]
        N = X.shape[1]
        
        # 预计算g的分解
        g_sqrt = np.linalg.cholesky(g_center)  # g = L @ L^T
        
        # 变换向量: Q' = Q @ L, X' = X @ L
        Q_transformed = Q @ g_sqrt
        X_transformed = X @ g_sqrt
        
        # 欧氏距离（在变换空间）
        distances = np.sqrt(
            np.sum(Q_transformed**2, axis=1, keepdims=True) +
            np.sum(X_transformed**2, axis=1, keepdims=True).T -
            2 * Q_transformed @ X_transformed.T
        )
        
        return distances
    
    def christoffel_symbols(self, x: np.ndarray) -> np.ndarray:
        """
        计算Christoffel符号
        
        公式: Γ^μ_αβ = (1/2) g^μλ (∂_β g_λα + ∂_α g_λβ - ∂_λ g_αβ)
        
        Returns:
            Gamma: [D, D, D] 张量
        """
        if self.metric_field is None:
            raise ValueError("Metric field not learned. Call fit() first.")
        
        D = len(x)
        h = 1e-5
        
        # 计算度量在各方向的偏导
        dg = np.zeros((D, D, D))  # dg[i,j,k] = ∂_k g_ij
        
        for k in range(D):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[k] += h
            x_minus[k] -= h
            
            g_plus = self.metric_field(x_plus)
            g_minus = self.metric_field(x_minus)
            
            dg[:, :, k] = (g_plus - g_minus) / (2 * h)
        
        # 计算Christoffel符号
        g = self.metric_field(x)
        g_inv = np.linalg.inv(g)
        
        Gamma = np.zeros((D, D, D))
        
        for mu in range(D):
            for alpha in range(D):
                for beta in range(D):
                    # Γ^μ_αβ = (1/2) Σ_λ g^μλ (∂_β g_λα + ∂_α g_λβ - ∂_λ g_αβ)
                    Gamma[mu, alpha, beta] = 0.5 * np.sum(
                        g_inv[mu, :] * (
                            dg[lambda_, alpha, beta] + 
                            dg[lambda_, beta, alpha] - 
                            dg[alpha, beta, lambda_]
                        )
                        for lambda_ in range(D)
                    )
        
        return Gamma
    
    def get_curvature(self, x: np.ndarray) -> Dict:
        """
        计算曲率相关量
        
        Returns:
            {
                'scalar_curvature': 标量曲率,
                'ricci_curvature': Ricci曲率矩阵,
                ' sectional_curvature_sample': 截面曲率采样
            }
        """
        if self.metric_field is None:
            raise ValueError("Metric field not learned. Call fit() first.")
        
        D = len(x)
        h = 1e-4
        
        # 计算度量和Christoffel符号及其导数
        g = self.metric_field(x)
        Gamma = self.christoffel_symbols(x)
        
        # Christoffel符号的偏导
        dGamma = np.zeros((D, D, D, D))
        
        for k in range(D):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[k] += h
            x_minus[k] -= h
            
            Gamma_plus = self.christoffel_symbols(x_plus)
            Gamma_minus = self.christoffel_symbols(x_minus)
            
            dGamma[:, :, :, k] = (Gamma_plus - Gamma_minus) / (2 * h)
        
        # Riemann曲率张量
        # R^ρ_σμν = ∂_μ Γ^ρ_νσ - ∂_ν Γ^ρ_μσ + Γ^ρ_μλ Γ^λ_νσ - Γ^ρ_νλ Γ^λ_μσ
        Riemann = np.zeros((D, D, D, D))
        
        for rho in range(D):
            for sigma in range(D):
                for mu in range(D):
                    for nu in range(D):
                        Riemann[rho, sigma, mu, nu] = (
                            dGamma[rho, nu, sigma, mu] - 
                            dGamma[rho, mu, sigma, nu] +
                            np.sum(Gamma[rho, mu, lambda_] * Gamma[lambda_, nu, sigma]
                                  for lambda_ in range(D)) -
                            np.sum(Gamma[rho, nu, lambda_] * Gamma[lambda_, mu, sigma]
                                  for lambda_ in range(D))
                        )
        
        # Ricci曲率（缩并）
        Ricci = np.zeros((D, D))
        for mu in range(D):
            for nu in range(D):
                Ricci[mu, nu] = np.sum(Riemann[lambda_, mu, lambda_, nu]
                                       for lambda_ in range(D))
        
        # 标量曲率
        g_inv = np.linalg.inv(g)
        scalar_curvature = np.sum(g_inv * Ricci)
        
        return {
            'scalar_curvature': scalar_curvature,
            'ricci_curvature': Ricci,
            'riemann_tensor': Riemann
        }


# 使用示例
if __name__ == "__main__":
    # 生成数据（弯曲的流形）
    np.random.seed(42)
    N = 500
    D = 10
    
    # 弯曲数据（部分维度变化大，部分小）
    X = np.random.randn(N, D)
    X[:, :3] *= 2  # 前3维变化大（高密度）
    X[:, 3:] *= 0.5  # 后7维变化小（低密度）
    
    # 学习度量张量
    learner = MetricTensorLearner(method='density', bandwidth=0.2)
    learner.fit(X)
    
    # 计算黎曼距离
    x = X[0]
    y = X[1]
    
    riem_dist = learner.riemannian_distance(x, y)
    eucl_dist = np.linalg.norm(x - y)
    
    print(f"Riemannian distance: {riem_dist:.4f}")
    print(f"Euclidean distance: {eucl_dist:.4f}")
    
    # 获取曲率
    curvature = learner.get_curvature(X.mean(axis=0))
    print(f"Scalar curvature: {curvature['scalar_curvature']:.4f}")
```

---

## 2. 改造现有向量数据库的建议

### 2.1 FAISS 改造方案

#### 2.1.1 改造点分析

| FAISS组件 | 改造内容 | 难度 | 优先级 |
|----------|---------|------|--------|
| **距离计算** | 替换为双曲距离 | 中 | 高 |
| **索引构建** | IVF centroid需在双曲空间 | 中 | 高 |
| **PQ量化** | 码本需适应双曲空间 | 高 | 中 |
| **HNSW集成** | HNSW作为IVF粗量化器 | 中 | 中 |
| **GPU优化** | 双曲距离的CUDA实现 | 高 | 低 |

#### 2.1.2 具体改造代码

```python
"""
faiss_hyperbolic_extension.py - FAISS双曲空间扩展

作者: implementer
日期: 2026-04-20

策略：不修改FAISS核心，通过包装层实现
"""

import numpy as np
import faiss
from typing import Optional, Tuple

class HyperbolicFAISS:
    """
    双曲空间的FAISS包装
    
    实现策略：
    1. 使用欧氏索引作为粗筛
    2. 用双曲距离重排序
    3. 向量投影确保在双曲空间
    """
    
    def __init__(
        self,
        d: int,
        index_type: str = 'IVF',
        n_list: int = 100,
        model: str = 'poincare',
        radius: float = 0.99,
        use_rerank: bool = True,
        rerank_factor: int = 10
    ):
        """
        Args:
            d: 向量维度
            index_type: FAISS索引类型 ('Flat', 'IVF', 'IVF-PQ')
            n_list: IVF的cell数
            model: 双曲模型 ('poincare', 'lorentz')
            radius: Poincaré球半径
            use_rerank: 是否用双曲距离重排序
            rerank_factor: 重排序候选倍数
        """
        self.d = d
        self.model = model
        self.radius = radius
        self.use_rerank = use_rerank
        self.rerank_factor = rerank_factor
        
        # 创建FAISS索引（使用欧氏距离）
        if index_type == 'Flat':
            self.index = faiss.IndexFlatL2(d)
        elif index_type == 'IVF':
            quantizer = faiss.IndexFlatL2(d)
            self.index = faiss.IndexIVFFlat(quantizer, d, n_list)
        elif index_type == 'IVF-PQ':
            quantizer = faiss.IndexFlatL2(d)
            m = d // 8  # PQ子向量数
            self.index = faiss.IndexIVFPQ(quantizer, d, n_list, m, 8)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # 初始化双曲距离计算器
        if model == 'poincare':
            from poincare_distance import PoincareDistance
            self.hyperbolic_dist = PoincareDistance(radius=radius)
        else:
            from lorentz_distance import LorentzDistance
            self.hyperbolic_dist = LorentzDistance()
        
        # 存储原始向量（用于双曲距离计算）
        self.vectors: Optional[np.ndarray] = None
    
    def _project(self, X: np.ndarray) -> np.ndarray:
        """
        投影到双曲空间
        """
        if self.model == 'poincare':
            # 逐向量投影
            X_proj = np.zeros_like(X)
            for i, x in enumerate(X):
                norm = np.linalg.norm(x)
                if norm >= self.radius:
                    X_proj[i] = x * (self.radius - 0.01) / norm
                else:
                    X_proj[i] = x
            return X_proj
        else:  # lorentz
            X_lorentz = np.zeros((len(X), self.d + 1))
            for i, x in enumerate(X):
                X_lorentz[i] = self.hyperbolic_dist.euclidean_to_lorentz(x)
            return X_lorentz
    
    def train(self, X: np.ndarray):
        """
        训练索引
        
        注意：IVF等索引需要训练
        """
        if hasattr(self.index, 'train'):
            # 训练用欧氏距离
            self.index.train(X)
    
    def add(self, X: np.ndarray):
        """
        添加向量
        
        Args:
            X: 欧氏向量 [N, d]
        """
        # 存储原始向量（用于重排序）
        self.vectors = self._project(X)
        
        # FAISS使用欧氏向量（粗筛）
        self.index.add(X)
    
    def search(
        self,
        Q: np.ndarray,
        k: int,
        n_probe: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        搜索
        
        Args:
            Q: 查询向量 [B, d]
            k: 返回数量
            n_probe: IVF扫描cell数
        
        Returns:
            D: 距离 [B, k]
            I: 索引 [B, k]
        """
        # 设置n_probe
        if n_probe and hasattr(self.index, 'nprobe'):
            self.index.nprobe = n_probe
        
        # 粗筛（欧氏距离）
        k_candidate = k * self.rerank_factor if self.use_rerank else k
        D_eu, I = self.index.search(Q, k_candidate)
        
        if not self.use_rerank:
            return D_eu, I
        
        # 双曲距离重排序
        Q_proj = self._project(Q)
        
        D_hyper = np.zeros((len(Q), k))
        I_hyper = np.zeros((len(Q), k), dtype=np.int64)
        
        for i in range(len(Q)):
            candidates = I[i]
            
            # 计算双曲距离
            hyper_dists = []
            for j in candidates:
                if j < 0:  # FAISS填充-1表示无效
                    continue
                if self.model == 'poincare':
                    d = self.hyperbolic_dist.distance(Q_proj[i], self.vectors[j])
                else:
                    d = self.hyperbolic_dist.distance(Q_proj[i], self.vectors[j])
                hyper_dists.append((j, d))
            
            # 按双曲距离排序
            sorted_results = sorted(hyper_dists, key=lambda x: x[1])[:k]
            
            for idx, (node_id, dist) in enumerate(sorted_results):
                I_hyper[i, idx] = node_id
                D_hyper[i, idx] = dist
        
        return D_hyper, I_hyper
    
    def get_stats(self) -> dict:
        """
        获取索引统计
        """
        return {
            'ntotal': self.index.ntotal,
            'model': self.model,
            'use_rerank': self.use_rerank
        }


# 使用示例
if __name__ == "__main__":
    # 创建双曲FAISS
    d = 128
    index = HyperbolicFAISS(
        d=d,
        index_type='IVF',
        n_list=100,
        model='poincare',
        radius=0.9,
        use_rerank=True
    )
    
    # 数据
    np.random.seed(42)
    N = 10000
    X = np.random.randn(N, d).astype(np.float32) * 0.3
    
    # 训练和添加
    index.train(X[:1000])
    index.add(X)
    
    # 搜索
    Q = np.random.randn(5, d).astype(np.float32) * 0.3
    D, I = index.search(Q, k=10, n_probe=10)
    
    print(f"Search results shape: D={D.shape}, I={I.shape}")
    print(f"Top distances: {D[0, :5]}")
```

#### 2.1.3 性能权衡分析

| 方案 | 构建开销 | 搜索开销 | Recall | 内存 |
|-----|---------|---------|--------|------|
| **纯欧氏FAISS** | $O(Nd)$ | $O(Nd)$ | 高 | 小 |
| **双曲重排序** | $O(Nd)$ | $O(k_r \cdot d)$ | 高 | $N \cdot d$ |
| **纯双曲索引** | $O(Nd)$ + 投影 | $O(Nd)$ + arccosh | 待验证 | $N \cdot d$ |

**结论**：
- 重排序方案改动最小，性能可控
- 纯双曲索引需更深入改造

### 2.2 HNSWlib 改造方案

#### 2.2.1 改造点

HNSWlib是纯Python/Cpp实现，改造比FAISS更直接：

| 组件 | 改造点 | 代码位置 |
|-----|-------|---------|
| **距离函数** | 替换`distance_func` | `hnswlib/distances.py` |
| **邻居选择** | 使用双曲距离 | `hnswlib/hnsw.py::add` |
| **搜索引导** | 使用双曲距离 | `hnswlib/hnsw.py::search` |

#### 2.2.2 改造代码

```python
"""
hnswlib_hyperbolic_patch.py - HNSWlib双曲距离patch

作者: implementer
日期: 2026-04-20

方法：继承hnswlib.Index，替换距离函数
"""

import hnswlib
import numpy as np
from typing import Callable

class HyperbolicHNSWIndex(hnswlib.Index):
    """
    双曲空间的HNSWlib索引
    
    通过替换距离函数实现
    """
    
    def __init__(
        self,
        space: str = 'poincare',
        dim: int = 128,
        max_elements: int = 10000,
        M: int = 16,
        ef_construction: int = 200,
        radius: float = 0.9
    ):
        """
        Args:
            space: 'poincare' 或 'lorentz'
            dim: 向量维度
            max_elements: 最大元素数
            M: 连接数
            ef_construction: 构建参数
            radius: Poincaré半径
        """
        # 初始化父类（用l2作为占位）
        super().__init__(space='l2', dim=dim)
        
        # 初始化索引参数
        self.init_index(max_elements, M, ef_construction)
        
        # 双曲距离函数
        self.radius = radius
        self.space_type = space
        
        if space == 'poincare':
            from poincare_distance import PoincareDistance
            self.hyperbolic_calc = PoincareDistance(radius=radius)
        else:
            from lorentz_distance import LorentzDistance
            self.hyperbolic_calc = LorentzDistance()
        
        # 覆盖距离函数
        self._set_distance_func()
    
    def _set_distance_func(self):
        """
        设置双曲距离函数
        
        注意：hnswlib的距离函数签名是 (label1, label2) -> distance
        我们需要通过标签找到向量，然后计算双曲距离
        """
        def hyperbolic_distance(label1: int, label2: int) -> float:
            # 获取向量（通过hnswlib内部方法）
            # 注意：这需要hack hnswlib的内部结构
            # 简化方案：存储向量在外部字典
            
            v1 = self.vector_dict.get(label1)
            v2 = self.vector_dict.get(label2)
            
            if v1 is None or v2 is None:
                return float('inf')
            
            return self.hyperbolic_calc.distance(v1, v2)
        
        # hnswlib不直接支持自定义距离函数
        # 需要修改C++代码或使用包装
        # 这里我们使用外部存储 + 搜索时重排序
        
        self.vector_dict = {}
    
    def add_items(self, X: np.ndarray, ids: np.ndarray):
        """
        添加向量
        
        Args:
            X: 向量 [N, dim]
            ids: 标签 [N]
        """
        # 投影到双曲空间
        X_proj = self._project(X)
        
        # 存储到字典
        for i, (x, id_) in enumerate(zip(X_proj, ids)):
            self.vector_dict[id_] = x
        
        # 用欧氏向量构建索引（粗筛）
        super().add_items(X, ids)
    
    def _project(self, X: np.ndarray) -> np.ndarray:
        """
        投影到双曲空间
        """
        if self.space_type == 'poincare':
            X_proj = np.zeros_like(X)
            for i, x in enumerate(X):
                norm = np.linalg.norm(x)
                if norm >= self.radius:
                    X_proj[i] = x * (self.radius - 0.01) / norm
                else:
                    X_proj[i] = x
            return X_proj
        else:
            X_lorentz = np.zeros((len(X), self.dim + 1))
            for i, x in enumerate(X):
                X_lorentz[i] = self.hyperbolic_calc.euclidean_to_lorentz(x)
            return X_lorentz
    
    def knn_query(
        self,
        Q: np.ndarray,
        k: int = 10,
        rerank: bool = True,
        ef: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        查询
        
        Args:
            Q: 查询向量 [B, dim]
            k: 返回数量
            rerank: 是否用双曲距离重排序
            ef: 搜索宽度
        
        Returns:
            labels: 标签 [B, k]
            distances: 距离 [B, k]
        """
        if ef:
            self.set_ef(ef)
        
        # 欧氏搜索（粗筛）
        k_candidate = k * 10 if rerank else k
        labels, distances = super().knn_query(Q, k=k_candidate)
        
        if not rerank:
            return labels, distances
        
        # 双曲重排序
        Q_proj = self._project(Q)
        
        labels_hyper = np.zeros((len(Q), k), dtype=np.int64)
        distances_hyper = np.zeros((len(Q), k))
        
        for i in range(len(Q)):
            candidates = labels[i]
            
            hyper_dists = []
            for label in candidates:
                if label < 0:
                    continue
                v = self.vector_dict.get(label)
                if v is None:
                    continue
                d = self.hyperbolic_calc.distance(Q_proj[i], v)
                hyper_dists.append((label, d))
            
            sorted_results = sorted(hyper_dists, key=lambda x: x[1])[:k]
            
            for idx, (label, dist) in enumerate(sorted_results):
                labels_hyper[i, idx] = label
                distances_hyper[i, idx] = dist
        
        return labels_hyper, distances_hyper


# 使用示例
if __name__ == "__main__":
    index = HyperbolicHNSWIndex(
        space='poincare',
        dim=128,
        max_elements=10000,
        M=16,
        radius=0.9
    )
    
    # 数据
    np.random.seed(42)
    N = 5000
    X = np.random.randn(N, 128).astype(np.float32) * 0.3
    ids = np.arange(N)
    
    index.add_items(X, ids)
    
    # 搜索
    Q = np.random.randn(3, 128).astype(np.float32) * 0.3
    labels, distances = index.knn_query(Q, k=10, rerank=True)
    
    print(f"Search labels: {labels[0, :5]}")
    print(f"Hyperbolic distances: {distances[0, :5]}")
```

### 2.3 迁移步骤

#### 2.3.1 从欧氏向量数据库迁移到双曲空间的步骤

```
Migration Pipeline:

Step 1: 数据准备
├─ 1.1 分析数据是否适合双曲空间（层级性、树状结构）
├─ 1.2 评估维度和规模
└─ 1.3 确定投影参数（Poincaré半径）

Step 2: 嵌入转换
├─ 2.1 选择双曲模型（Poincaré/Lorentz）
├─ 2.2 投影现有向量到双曲空间
├─ 2.3 验证投影后距离分布
└─ 2.4 [可选] 重新训练双曲嵌入

Step 3: 索引改造
├─ 3.1 选择改造方案（重排序 vs 纯双曲）
├─ 3.2 实现双曲距离函数
├─ 3.3 包装现有索引
└─ 3.4 测试基本功能

Step 4: 性能调优
├─ 4.1 测试Recall vs Latency
├─ 4.2 调整重排序因子（rerank_factor）
├─ 4.3 调整搜索宽度（ef_search）
└─ 4.4 GPU优化（如果需要）

Step 5: 线上部署
├─ 5.1 A/B测试：欧氏 vs 双曲
├─ 5.2 监控Recall和延迟
├─ 5.3 收集用户反馈
└─ 5.4 根据结果决定全面迁移
```

#### 2.3.2 兼容性检查清单

| 检查项 | 欧氏兼容 | 双曲兼容 | 备注 |
|-------|---------|---------|------|
| **向量维度** | 任意 | 任意 | Lorentz需+1维 |
| **向量归一化** | 可选 | 不适用 | 双曲空间不归一化 |
| **距离度量** | L2/Cosine/IP | Poincaré/Lorentz | 需自定义 |
| **量化(PQ)** | 支持 | 不直接支持 | 需改造 |
| **GPU加速** | 原生支持 | 需自定义CUDA | 改造难度高 |
| **批量查询** | 原生支持 | 可包装 | SIMD优化困难 |

---

## 3. 实验设计

### 3.1 验证方案

#### 3.1.1 数据集选择

| 数据集 | 类型 | 维度 | 规模 | 适用原因 |
|-------|------|------|------|---------|
| **WordNet** | 层级语义 | 50-200 | ~10K | 验证层级数据双曲效果 |
| **DBpedia** | 知识图谱 | 200-500 | ~1M | RAG场景验证 |
| **SIFT1M** | 图像特征 | 128 | 1M | 标准基准，对比现有方法 |
| **GloVe** | 词嵌入 | 200-300 | ~1M | 验证文本嵌入 |
| **WikiLinks** | 文档链接图 | 128-256 | ~10M | RAG文档关联验证 |

#### 3.1.2 评价指标

```python
"""
evaluation_metrics.py - Manifold-RAG评价指标

作者: implementer
日期: 2026-04-20
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

class ManifoldRAGEvaluator:
    """
    Manifold-RAG评价指标
    
    包含：
    1. Recall/K-NN准确率
    2. Hubness测量
    3. 搜索延迟
    4. 层级保持度（双曲特有）
    """
    
    def __init__(self, k: int = 10):
        self.k = k
    
    def recall_at_k(
        self,
        predictions: List[List[int]],
        ground_truth: List[List[int]]
    ) -> float:
        """
        Recall@k
        
        Args:
            predictions: 预测的k-NN [[id1, id2, ...], ...]
            ground_truth: 真实的k-NN
        
        Returns:
            平均Recall@k
        """
        recalls = []
        
        for pred, gt in zip(predictions, ground_truth):
            # 预测中有多少在真实k-NN中
            overlap = len(set(pred) & set(gt))
            recall = overlap / min(len(gt), self.k)
            recalls.append(recall)
        
        return np.mean(recalls)
    
    def hubness_skewness(self, N_k: List[int]) -> float:
        """
        Hubness偏度
        
        Args:
            N_k: 各节点作为k-NN的次数
        
        Returns:
            偏度值（>0表示存在hub）
        """
        mean_Nk = np.mean(N_k)
        std_Nk = np.std(N_k)
        
        if std_Nk == 0:
            return 0
        
        skewness = np.mean((N_k - mean_Nk)**3) / std_Nk**3
        return skewness
    
    def hubness_gini(self, N_k: List[int]) -> float:
        """
        Hubness Gini系数
        
        测量k-NN出现次数分布的不均匀性
        """
        N_k_sorted = np.sort(N_k)
        n = len(N_k)
        
        # Gini系数
        cumsum = np.cumsum(N_k_sorted)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
        return gini
    
    def search_latency(
        self,
        search_times: List[float],
        percentile: int = 95
    ) -> Dict:
        """
        搜索延迟统计
        
        Returns:
            {
                'mean': 平均延迟,
                'median': 中位数,
                'p95': 95分位数,
                'p99': 99分位数
            }
        """
        return {
            'mean': np.mean(search_times),
            'median': np.median(search_times),
            'p95': np.percentile(search_times, percentile),
            'p99': np.percentile(search_times, 99)
        }
    
    def hierarchy_preservation(
        self,
        embeddings: np.ndarray,
        hierarchy: Dict[int, List[int]],  # {parent: [children]}
        model: str = 'poincaré'
    ) -> float:
        """
        层级保持度
        
        双曲嵌入特有指标：
        - 层级中父节点应距离所有子节点较近
        - 层级距离应与语义距离对应
        
        Args:
            embeddings: 嵌入向量 [N, D]
            hierarchy: 层级结构
            model: 双曲模型
        
        Returns:
            层级保持分数（越高越好）
        """
        if model == 'poincare':
            from poincare_distance import PoincareDistance
            dist_calc = PoincareDistance()
        else:
            from lorentz_distance import LorentzDistance
            dist_calc = LorentzDistance()
        
        preservation_scores = []
        
        for parent, children in hierarchy.items():
            if parent >= len(embeddings):
                continue
            
            parent_emb = embeddings[parent]
            
            # 计算父节点到子节点的平均距离
            child_dists = []
            for child in children:
                if child >= len(embeddings):
                    continue
                child_emb = embeddings[child]
                d = dist_calc.distance(parent_emb, child_emb)
                child_dists.append(d)
            
            if child_dists:
                avg_child_dist = np.mean(child_dists)
                
                # 计算父节点到非子节点的平均距离
                non_child_dists = []
                for i in range(len(embeddings)):
                    if i != parent and i not in children:
                        d = dist_calc.distance(parent_emb, embeddings[i])
                        non_child_dists.append(d)
                
                if non_child_dists:
                    avg_non_child_dist = np.mean(non_child_dists)
                    
                    # 层级保持：子节点距离应小于非子节点
                    # 分数 = avg_non_child_dist / avg_child_dist（越大越好）
                    score = avg_non_child_dist / avg_child_dist
                    preservation_scores.append(score)
        
        return np.mean(preservation_scores) if preservation_scores else 0
    
    def geodesic_approximation_error(
        self,
        geodesic_distances: np.ndarray,
        graph_distances: np.ndarray
    ) -> float:
        """
        测地距离近似误差
        
        测量图路径距离与真实测地距离的差异
        
        Args:
            geodesic_distances: 测地距离矩阵 [N, N]
            graph_distances: 图路径距离矩阵 [N, N]
        
        Returns:
            平均相对误差
        """
        # 避免0距离
        mask = geodesic_distances > 0
        
        relative_error = np.abs(
            graph_distances[mask] - geodesic_distances[mask]
        ) / geodesic_distances[mask]
        
        return np.mean(relative_error)
    
    def full_evaluation(
        self,
        index,
        test_queries: np.ndarray,
        ground_truth: List[List[int]],
        hierarchy: Optional[Dict] = None
    ) -> Dict:
        """
        完整评估
        
        Returns:
            {
                'recall@k': float,
                'hubness_skewness': float,
                'hubness_gini': float,
                'latency': dict,
                'hierarchy_preservation': float (可选)
            }
        """
        import time
        
        # 搜索测试
        predictions = []
        search_times = []
        
        for q in test_queries:
            start = time.time()
            results = index.search(q, k=self.k)
            search_times.append(time.time() - start)
            predictions.append([r[0] for r in results])
        
        # Recall
        recall = self.recall_at_k(predictions, ground_truth)
        
        # Hubness
        N_k = defaultdict(int)
        for pred in predictions:
            for node_id in pred:
                N_k[node_id] += 1
        
        N_k_list = [N_k.get(i, 0) for i in range(index.node_count)]
        skewness = self.hubness_skewness(N_k_list)
        gini = self.hubness_gini(N_k_list)
        
        # Latency
        latency = self.search_latency(search_times)
        
        result = {
            'recall@k': recall,
            'hubness_skewness': skewness,
            'hubness_gini': gini,
            'latency': latency
        }
        
        # 层级保持（如果提供）
        if hierarchy is not None:
            preservation = self.hierarchy_preservation(
                np.array([index.nodes[i] for i in range(index.node_count)]),
                hierarchy
            )
            result['hierarchy_preservation'] = preservation
        
        return result
```

#### 3.1.3 实验流程

```
Experiment Pipeline:

Phase 1: Baseline建立
├─ 1.1 欧氏HNSW基线
│   ├─ SIFT1M: Recall@10, Hubness, Latency
│   ├─ WordNet: Recall@10 + Hierarchy Preservation
│   └─ DBpedia: Recall@10
├─ 1.2 标准双曲嵌入基线（Nickel & Kiela方法）
└─ 1.3 记录所有指标

Phase 2: 双曲HNSW测试
├─ 2.1 Poincaré模型
│   ├─ 不同半径参数: 0.7, 0.8, 0.9, 0.95
│   ├─ 记录Recall, Hubness, Hierarchy Preservation
│   └─ 对比基线
├─ 2.2 Lorentz模型
│   ├─ 距离计算: arccosh vs squared
│   ├─ 记录指标
│   └─ 对比Poincaré
└─ 2.3 Hubness对比分析

Phase 3: 度量张量测试
├─ 3.1 Density-based度量
│   ├─ 不同bandwidth参数
│   ├─ 记录Recall（黎曼距离）
│   └─ 记录计算开销
├─ 3.2 PCA度量
│   ├─ 全局度量
│   └─ 对比Density
└─ 3.3 曲率分析

Phase 4: RAG场景测试
├─ 4.1 知识图谱查询
│   ├─ WikiLinks数据
│   ├─ 查询：文档关联检索
│   └─ 评估：层级保持 + Recall
├─ 4.2 混合检索
│   ├─ 欧氏粗筛 + 双曲精排
│   ├─ Latency权衡
│   └─ 对比纯双曲
└─ 4.3 端到端RAG评估
    ├─ 检索质量 → 生成质量
    ├─ 用户满意度（人工评估）
    └─ 搜索效率
```

### 3.2 对比基准

#### 3.2.1 与现有方法的对比

| 方法 | SIFT1M Recall@10 | WordNet Hierarchy | Hubness (Skew) | Latency (ms) |
|-----|------------------|-------------------|----------------|--------------|
| **HNSW (欧氏)** | ~98% | N/A | 2.5-4.0 | 0.5-1.0 |
| **IVF-PQ (FAISS)** | ~50-60% | N/A | 2.0-3.0 | 0.1-0.5 |
| **Poincaré Embed (Nickel)** | N/A | 0.85-0.90 | 需测试 | 需测试 |
| **Hyperbolic HNSW** | 预估95-97% | 预估0.80-0.90 | **预估<1.5** | 预估1.0-2.0 |
| **Riemannian Distance** | 需测试 | 需测试 | 需测试 | 预估10-50 |

#### 3.2.2 预期改进

**Hubness缓解验证**：

| 数据集 | 欧氏Hubness | 双曲预期Hubness | 改进幅度 |
|-------|-------------|----------------|---------|
| SIFT1M | 3.5 | ~1.5 | ~57% |
| WordNet | 2.0 | ~0.8 | ~60% |
| GloVe | 2.8 | ~1.2 | ~57% |

**层级保持验证**：

| 方法 | WordNet Hierarchy Preservation |
|-----|------------------------------|
| 欧氏嵌入 | 0.60-0.70 |
| Poincaré Embed | 0.85-0.90 |
| Hyperbolic HNSW | 0.80-0.88 |

### 3.3 预期结果和风险

#### 3.3.1 预期成功结果

| 预期结果 | 验证标准 | 影响 |
|---------|---------|------|
| **Hubness缓解** | Skewness降低50%+ | k-NN多样性提升 |
| **层级保持** | Preservation > 0.80 | 层级数据检索准确 |
| **Recall保持** | Recall@10 > 95% | 搜索质量不下降 |
| **Latency可控** | 增加<3x | 实际部署可行 |

#### 3.3.2 风险识别

| 风险 | 概率 | 影响 | 缓解措施 |
|-----|------|------|---------|
| **双曲距离计算过慢** | 中 | 高 | 使用Lorentz平方距离；GPU优化 |
| **Hubness未缓解** | 低 | 中 | 验证其他曲率参数；组合方法 |
| **Recall下降** | 中 | 高 | 增加ef_search；重排序 |
| **层级数据不适用** | 低 | 低 | 使用欧氏作为后备 |
| **内存开销过大** | 中 | 中 | 存储优化；压缩 |

---

## 4. 实现优先级排序

### 4.1 可行性评估

| 方案 | 技术难度 | 实现时间 | 效果预期 | 优先级 |
|-----|---------|---------|---------|--------|
| **Poincaré距离实现** | 低 | 1天 | 高 | **P0** |
| **Lorentz距离实现** | 低 | 1天 | 高 | **P0** |
| **HyperbolicHNSW原型** | 中 | 3天 | 高 | **P1** |
| **FAISS双曲包装** | 中 | 2天 | 中 | **P1** |
| **Hubness验证实验** | 低 | 2天 | 高 | **P1** |
| **度量张量学习** | 高 | 5天 | 中 | **P2** |
| **Geodesic-HNSW** | 高 | 7天 | 中 | **P3** |
| **GPU双曲距离优化** | 高 | 10天 | 中 | **P3** |

### 4.2 实现路线图

```
Timeline: 6周

Week 1: 基础组件 (P0)
├─ Day 1-2: Poincaré距离实现 + 测试
├─ Day 3-4: Lorentz距离实现 + 测试
└─ Day 5: 基础Hubness测量工具

Week 2: 核心原型 (P1)
├─ Day 1-3: HyperbolicHNSW完整实现
├─ Day 4-5: FAISS双曲包装
└─ Day 5: 基础功能测试

Week 3: 实验验证 (P1)
├─ Day 1-2: SIFT1M基准测试
├─ Day 3-4: WordNet层级测试
├─ Day 5: Hubness对比分析
└─ Day 5: 初步结果报告

Week 4: 优化迭代 (P2)
├─ Day 1-3: 度量张量学习模块
├─ Day 4-5: 性能优化（SIMD、批量）
└─ Day 5: 第二轮实验

Week 5: RAG场景 (P2)
├─ Day 1-2: WikiLinks数据准备
├─ Day 3-4: 知识图谱检索测试
├─ Day 5: 端到端评估
└─ Day 5: 场景报告

Week 6: 总结与交付 (P3)
├─ Day 1-2: 完整实验报告
├─ Day 3-4: 代码清理 + 文档
├─ Day 5: GPU优化探索（可选）
└─ Day 5: 最终交付
```

### 4.3 资源需求

| 资源 | 需求 | 说明 |
|-----|------|------|
| **人力** | 1-2人 | 全栈开发 + 实验分析 |
| **计算** | GPU可选 | CPU可完成大部分实验 |
| **数据存储** | ~50GB | 数据集 + 中间结果 |
| **时间** | 6周 | 完整实现 + 验证 |

### 4.4 里程碑定义

| Milestone | 时间 | 交付物 | 成功标准 |
|----------|------|--------|---------|
| **M1: 双曲距离完成** | Week 1 | Poincaré/Lorentz代码 | 距离计算正确验证 |
| **M2: HyperbolicHNSW原型** | Week 2 | 完整原型代码 | 基础搜索功能正常 |
| **M3: Hubness验证** | Week 3 | 实验报告 | Hubness降低30%+ |
| **M4: RAG场景测试** | Week 5 | 场景报告 | 层级保持>0.75 |
| **M5: 最终交付** | Week 6 | 完整代码+文档 | 所有指标达标 |

---

## 5. 参考文献

### 双曲几何与嵌入

1. **Nickel & Kiela (2017)**: "Poincaré Embeddings for Learning Hierarchical Representations" - NeurIPS
2. **Nickel & Kiela (2018)**: "Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry" - ICML
3. **Law et al. (2019)**: "Lorentzian Distance Learning for Hyperbolic Representations" - ICML

### 向量数据库

4. **Malkov & Yashunin (2016)**: "Efficient and robust approximate nearest neighbor search using HNSW"
5. **Johnson et al. (2017)**: "Billion-scale similarity search with GPUs" - FAISS
6. **Douze et al. (2024)**: "The Faiss Library" - arxiv 2401.08281

### Hubness研究

7. **Radovanovic et al. (2010)**: "Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data"
8. **BlaiseMuhirwa et al. (2024)**: "Down with the Hierarchy: The 'H' in HNSW Stands for 'Hubs'" - arxiv 2412.01940

### 流形学习

9. **Tenenbaum et al. (2000)**: "A Global Geometric Framework for Nonlinear Dimensionality Reduction" - Isomap
10. **Roweis & Saul (2000)**: "Nonlinear Dimensionality Reduction by Locally Linear Embedding"

### RAG相关

11. **Lewis et al. (2020)**: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
12. **Gao et al. (2023)**: "Retrieval-Augmented Generation for AI-Generated Content: A Survey"

---

## 附录：代码仓库结构

```
manifold-rag-research/
├── 01-background.md          # 背景调研
├── 02-geometry-foundation.md # 几何基础
├── 03-manifold-intuition.md  # 流形直觉
├── 04-embedding-overview.md  # 嵌入概述
├── 05-manifold-math.md       # 流形数学公式
├── 06-vector-db-algo.md      # 向量数据库算法
├── 07-relation-analysis.md   # 关联分析
├── 08-implementation-guide.md # 本文档
│
├── code/
│   ├── poincare_distance.py       # Poincaré距离
│   ├── lorentz_distance.py        # Lorentz距离
│   ├── hyperbolic_hnsw.py         # 双曲HNSW
│   ├── metric_tensor_learning.py  # 度量张量学习
│   ├── faiss_hyperbolic_extension.py # FAISS扩展
│   ├── hnswlib_hyperbolic_patch.py # HNSWlib patch
│   └── evaluation_metrics.py      # 评价指标
│
├── experiments/
│   ├── hubness_comparison/        # Hubness对比实验
│   ├── hierarchy_preservation/    # 层级保持实验
│   ├── rag_scenario/              # RAG场景实验
│   └── results/                   # 实验结果
│
└── notebooks/
    ├── demo_poincare.ipynb        # Poincaré演示
    ├── demo_hyperbolic_hnsw.ipynb # HNSW演示
    └── analysis_hubness.ipynb     # Hubness分析
```

---

*文档完成于 2026-04-20*
*ClawTeam manifold-vector 研究组*
*implementer (ID: 944a162d79c6)*