# 意图聚类与动态权重方案设计

> 版本: v1.0  
> 作者: methodology-designer  
> 日期: 2026-03-25  
> 团队: intent-clustering-team

---

## 1. 意图聚类模块设计

### 1.1 无监督意图聚类算法选择

#### 推荐方案：层次化聚类 + 在线增量更新

```
┌─────────────────────────────────────────────────────────────┐
│                    意图聚类架构                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   用户问题 ──► Embedding ──► 粗粒度聚类 ──► 细粒度聚类      │
│      │              │              │              │        │
│      │              ▼              ▼              ▼        │
│      │         语义向量        HDBSCAN        K-Means      │
│      │              │              │              │        │
│      └──────────────┴──────────────┴──────────────┘        │
│                           │                                 │
│                           ▼                                 │
│                    意图类别标签                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 算法组合

| 层级 | 算法 | 适用场景 | 参数建议 |
|-----|------|---------|---------|
| **粗粒度** | HDBSCAN | 自动发现簇数量，处理噪声点 | `min_cluster_size=10`, `min_samples=5` |
| **细粒度** | 在线K-Means | 快速增量更新，稳定聚类中心 | 动态k值，基于轮廓系数 |
| **语义增强** | Spectral Clustering | 发现非凸形状的意图簇 | affinity='cosine' |

#### 为什么选择这个组合？

1. **HDBSCAN**：
   - 无需预设簇数量（适合未知意图场景）
   - 自动识别噪声/异常问题
   - 对初始数据分布不敏感

2. **在线K-Means**：
   - 增量更新效率高，适合实时场景
   - 内存占用小
   - 聚类结果稳定可解释

3. **层次结构**：
   - 粗粒度聚类发现主要意图大类
   - 细粒度聚类在类内细化子意图
   - 支持意图层次导航

### 1.2 用户问题累积与聚类更新策略

#### 增量更新机制

```python
class IncrementalIntentCluster:
    def __init__(self):
        self.corpus_embeddings = []  # 累积的embedding池
        self.cluster_centers = {}    # 簇中心缓存
        self.cluster_version = 0     # 聚类版本号
        self.update_threshold = 50   # 触发更新的新问题阈值
        
    def add_question(self, question, embedding):
        """添加新问题"""
        self.corpus_embeddings.append({
            'text': question,
            'embedding': embedding,
            'timestamp': time.now()
        })
        
        # 检查是否需要触发更新
        if len(self.corpus_embeddings) % self.update_threshold == 0:
            self._trigger_recluster()
    
    def _trigger_recluster(self):
        """触发重新聚类"""
        self.cluster_version += 1
        # 全量重新聚类（后台任务）
        # 或增量更新簇中心
```

#### 更新策略

| 策略 | 触发条件 | 更新方式 | 延迟 |
|-----|---------|---------|-----|
| **实时微更新** | 每个新问题 | 在线K-Means单点更新 | <100ms |
| **定期小更新** | 累积50个新问题 | K-Means局部优化 | <5s |
| **周期大更新** | 每24小时 | HDBSCAN全量重聚类 | 后台 |
| **触发式更新** | 检测到新意图模式 | 混合策略 | 按需 |

### 1.3 实体/关系数据辅助聚类

#### 实体增强的聚类方法

```
┌────────────────────────────────────────────────────────────┐
│               实体增强聚类流程                              │
├────────────────────────────────────────────────────────────┤
│                                                            │
│   问题文本 ──► 实体识别 ──► 实体类型权重                   │
│       │           │              │                         │
│       │           ▼              ▼                         │
│       │      PER/ORG/LOC    实体向量增强                   │
│       │           │              │                         │
│       ▼           ▼              ▼                         │
│   语义向量 = α * sentence_embedding + β * entity_features  │
│                                                            │
│   其中: α + β = 1, 实体特征来自图谱嵌入                     │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

#### 实体类型权重映射

| 实体类型 | 权重系数 | 说明 |
|---------|---------|------|
| PER (人物) | 0.3 | 用户画像、偏好相关 |
| ORG (机构) | 0.25 | 组织相关意图 |
| LOC (地点) | 0.2 | 地理相关查询 |
| PROD (产品) | 0.15 | 产品咨询意图 |
| MISC (其他) | 0.1 | 通用实体 |

#### 关系数据利用

```python
def relation_enhanced_similarity(q1_embedding, q2_embedding, q1_entities, q2_entities, knowledge_graph):
    """关系增强的相似度计算"""
    
    # 1. 语义相似度
    semantic_sim = cosine_similarity(q1_embedding, q2_embedding)
    
    # 2. 实体重叠度
    entity_overlap = jaccard_similarity(q1_entities, q2_entities)
    
    # 3. 关系路径相似度（图谱中的最短路径）
    relation_paths = knowledge_graph.find_paths(q1_entities, q2_entities)
    relation_sim = compute_path_similarity(relation_paths)
    
    # 加权融合
    final_sim = (
        0.5 * semantic_sim +
        0.3 * entity_overlap +
        0.2 * relation_sim
    )
    
    return final_sim
```

---

## 2. 动态权重生成模块设计

### 2.1 用户Feedback → 奖励信号

#### Feedback类型与奖励映射

```
┌─────────────────────────────────────────────────────────────┐
│                  Feedback → 奖励信号转换                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   显式Feedback:                                             │
│   ┌──────────────────┬────────────────┬────────────────┐   │
│   │     行为         │    奖励值      │     说明        │   │
│   ├──────────────────┼────────────────┼────────────────┤   │
│   │   👍 点赞        │     +1.0       │   正向强反馈    │   │
│   │   👎 踩/踩       │     -1.0       │   负向强反馈    │   │
│   │   收藏/保存      │     +0.8       │   隐性正向      │   │
│   │   分享/转发      │     +0.9       │   高价值反馈    │   │
│   │   举报          │     -2.0        │   强负反馈      │   │
│   └──────────────────┴────────────────┴────────────────┘   │
│                                                             │
│   隐式Feedback:                                             │
│   ┌──────────────────┬────────────────┬────────────────┐   │
│   │     行为         │    奖励值      │     说明        │   │
│   ├──────────────────┼────────────────┼────────────────┤   │
│   │   停留时长>30s   │     +0.3       │   内容有价值    │   │
│   │   停留时长<5s    │     -0.3       │   内容不相关    │   │
│   │   复制内容      │     +0.5        │   内容有用      │   │
│   │   继续追问      │     +0.2/-0.5   │   需细化场景    │   │
│   │   重新提问      │     -0.4        │   答案不满意    │   │
│   │   会话中断      │     -0.2        │   轻度负向      │   │
│   └──────────────────┴────────────────┴────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 奖励信号累积公式

```python
def compute_cumulative_reward(feedback_list, decay_factor=0.95):
    """计算累积奖励（时间衰减）"""
    total_reward = 0
    for i, fb in enumerate(reversed(feedback_list)):
        time_weight = decay_factor ** i  # 近期权重高
        total_reward += fb.reward * time_weight
    return total_reward
```

### 2.2 权重与意图类别映射

#### 权重矩阵设计

```
              意图类别
           ┌───┬───┬───┬───┬───┐
           │ C1│ C2│ C3│ C4│ C5│  (C=Cluster)
    ┌──────┼───┼───┼───┼───┼───┤
    │ 节点1 │0.3│0.1│0.4│0.1│0.1│
知  │ 节点2 │0.2│0.5│0.1│0.1│0.1│
识  │ 节点3 │0.1│0.1│0.3│0.4│0.1│
图  │ 节点4 │0.4│0.2│0.1│0.1│0.2│
谱  │ ...  │...│...│...│...│...│
节  └──────┴───┴───┴───┴───┴───┘
点        ↑
          └─── 每个节点对不同意图的权重
```

#### 权重解释

- **行向量**：每个知识图谱节点对不同意图类别的相关度权重
- **列向量**：每个意图类别下，各知识节点的重要性排序
- **归一化**：每行权重和为1（Softmax）

### 2.3 权重更新机制

#### 在线学习更新策略

```python
class DynamicWeightUpdater:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.weights = {}  # {node_id: {cluster_id: weight}}
        self.momentum_cache = {}  # 动量缓存
        self.lr = learning_rate
        self.momentum = momentum
        
    def update(self, node_id, cluster_id, reward):
        """基于奖励信号更新权重"""
        
        current_weight = self.weights.get(node_id, {}).get(cluster_id, 0.5)
        
        # 计算梯度（奖励方向）
        gradient = reward * self.lr
        
        # 动量更新
        velocity = self.momentum_cache.get(node_id, {}).get(cluster_id, 0)
        velocity = self.momentum * velocity + gradient
        self.momentum_cache.setdefault(node_id, {})[cluster_id] = velocity
        
        # 更新权重
        new_weight = current_weight + velocity
        
        # 归一化（保持概率分布）
        self._normalize(node_id)
        
    def _normalize(self, node_id):
        """对节点权重进行Softmax归一化"""
        weights = self.weights.get(node_id, {})
        exp_weights = {k: np.exp(v) for k, v in weights.items()}
        total = sum(exp_weights.values())
        self.weights[node_id] = {k: v/total for k, v in exp_weights.items()}
```

#### 更新触发机制

| 触发条件 | 更新策略 | 频率限制 |
|---------|---------|---------|
| 单次强正反馈（点赞） | 立即更新，权重+0.1 | 无 |
| 单次强负反馈（踩） | 立即更新，权重-0.1 | 无 |
| 累积隐式反馈 | 批量更新，权重±0.05 | 每小时 |
| 周期性重校准 | 全量权重归一化 | 每天 |
| 异常检测触发 | 权重衰减，恢复默认 | 按需 |

---

## 3. 召回策略设计

### 3.1 语义相似性 + 意图类别双路召回

#### 双路召回架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      双路召回架构                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   用户问题                                                       │
│       │                                                         │
│       ├─────────────────┬───────────────────┐                  │
│       ▼                 ▼                   ▼                  │
│   ┌───────┐        ┌───────┐          ┌───────┐              │
│   │语义召回│        │意图召回│          │混合召回│              │
│   │(Path A)│        │(Path B)│          │(Path C)│              │
│   └───┬───┘        └───┬───┘          └───┬───┘              │
│       │                │                   │                   │
│       ▼                ▼                   ▼                   │
│   相似度Top-K      意图类别Top-K       融合排序Top-K          │
│       │                │                   │                   │
│       └────────────────┴───────────────────┘                   │
│                        │                                       │
│                        ▼                                       │
│                   结果融合与排序                                 │
│                        │                                       │
│                        ▼                                       │
│                   最终召回结果                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 召回路径详解

**Path A: 语义相似性召回**
```python
def semantic_recall(query_embedding, corpus_embeddings, top_k=50):
    """基于语义相似度的召回"""
    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        corpus_embeddings
    )
    top_indices = np.argsort(similarities[0])[-top_k:][::-1]
    return top_indices, similarities[0][top_indices]
```

**Path B: 意图类别召回**
```python
def intent_recall(query_cluster, node_cluster_weights, top_k=50):
    """基于意图类别的召回"""
    # 获取该意图类别下权重最高的节点
    node_scores = {
        node_id: weights.get(query_cluster, 0)
        for node_id, weights in node_cluster_weights.items()
    }
    sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_nodes[:top_k]
```

**Path C: 混合召回**
```python
def hybrid_recall(query_embedding, query_cluster, corpus_embeddings, node_cluster_weights, alpha=0.6):
    """混合召回策略"""
    # 语义召回
    semantic_indices, semantic_scores = semantic_recall(query_embedding, corpus_embeddings)
    
    # 意图召回
    intent_nodes = intent_recall(query_cluster, node_cluster_weights)
    
    # 分数融合
    final_scores = {}
    for idx, score in zip(semantic_indices, semantic_scores):
        node_id = idx_to_node_id(idx)
        intent_score = dict(intent_nodes).get(node_id, 0)
        final_scores[node_id] = alpha * score + (1 - alpha) * intent_score
    
    # 排序返回
    return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
```

### 3.2 意图自适应的图谱数据召回

#### 自适应召回流程

```
┌───────────────────────────────────────────────────────────────┐
│                 意图自适应图谱召回                             │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│   Step 1: 意图识别                                           │
│   ┌─────────────────────────────────────────────────────┐   │
│   │   问题 → Embedding → 聚类匹配 → 意图类别概率分布     │   │
│   │                                                      │   │
│   │   例: [C1:0.1, C2:0.7, C3:0.15, C4:0.03, C5:0.02]   │   │
│   └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│   Step 2: 意图加权图谱遍历                                    │
│   ┌─────────────────────────────────────────────────────┐   │
│   │   从种子节点出发，按意图权重决定遍历方向              │   │
│   │                                                      │   │
│   │   边权重 = f(关系类型, 意图类别, 历史反馈)           │   │
│   │                                                      │   │
│   │   高权重路径优先遍历                                 │   │
│   └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│   Step 3: 候选节点排序                                       │
│   ┌─────────────────────────────────────────────────────┐   │
│   │   综合评分 = α*意图相关性 + β*路径置信度 + γ*新鲜度  │   │
│   │                                                      │   │
│   │   α, β, γ 可根据反馈动态调整                         │   │
│   └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│   Step 4: 返回Top-K结果                                      │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

#### 自适应参数调节

```python
class AdaptiveRecallParams:
    """自适应召回参数"""
    
    def __init__(self):
        self.alpha = 0.5  # 意图相关性权重
        self.beta = 0.3   # 路径置信度权重
        self.gamma = 0.2  # 新鲜度权重
        self.adaptation_rate = 0.1
        
    def adapt(self, feedback_history):
        """根据反馈历史自适应调整"""
        # 分析历史反馈模式
        positive_rate = sum(1 for fb in feedback_history if fb > 0) / len(feedback_history)
        
        # 如果正向反馈多，增加意图相关性权重
        if positive_rate > 0.7:
            self.alpha = min(0.7, self.alpha + self.adaptation_rate)
            self.beta = max(0.1, self.beta - self.adaptation_rate/2)
        
        # 如果负向反馈多，增加新鲜度探索
        elif positive_rate < 0.3:
            self.gamma = min(0.4, self.gamma + self.adaptation_rate)
            self.alpha = max(0.3, self.alpha - self.adaptation_rate/2)
```

---

## 4. RLHF适配设计

### 4.1 奖励函数设计

#### 多信号融合奖励函数

```
┌─────────────────────────────────────────────────────────────────┐
│                    奖励函数架构                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   R_total = R_explicit + R_implicit + R_temporal + R_quality    │
│                                                                 │
│   其中:                                                         │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │ R_explicit = Σ (显式反馈信号)                           │  │
│   │           = w_like * like + w_dislike * dislike         │  │
│   │           + w_save * save + w_share * share             │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │ R_implicit = Σ (隐式行为信号)                           │  │
│   │           = w_dwell * dwell_score                       │  │
│   │           + w_copy * copy_action                        │  │
│   │           + w_requery * requery_penalty                 │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │ R_temporal = 时间衰减因子                               │  │
│   │           = exp(-λ * time_since_feedback)               │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │ R_quality = 内容质量惩罚项                              │  │
│   │           = -penalty * (1 - safety_score)               │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 奖励函数实现

```python
class RewardFunction:
    def __init__(self):
        # 显式反馈权重
        self.w_like = 1.0
        self.w_dislike = -1.5
        self.w_save = 0.8
        self.w_share = 1.2
        
        # 隐式行为权重
        self.w_dwell = 0.3
        self.w_copy = 0.5
        self.w_requery = -0.4
        
        # 时间衰减
        self.lambda_decay = 0.01
        
        # 质量惩罚
        self.quality_penalty = 2.0
    
    def compute_reward(self, feedback_event):
        """计算单次反馈事件的奖励"""
        R_explicit = (
            self.w_like * feedback_event.like +
            self.w_dislike * feedback_event.dislike +
            self.w_save * feedback_event.save +
            self.w_share * feedback_event.share
        )
        
        R_implicit = (
            self.w_dwell * self._dwell_score(feedback_event.dwell_time) +
            self.w_copy * feedback_event.copy_action +
            self.w_requery * feedback_event.requery
        )
        
        R_temporal = np.exp(-self.lambda_decay * feedback_event.time_since)
        
        R_quality = -self.quality_penalty * (1 - feedback_event.safety_score)
        
        return R_explicit + R_implicit * R_temporal + R_quality
    
    def _dwell_score(self, dwell_time_seconds):
        """停留时间转换为分数"""
        if dwell_time_seconds < 5:
            return -0.3  # 太短，内容不相关
        elif dwell_time_seconds < 30:
            return 0.0   # 正常
        elif dwell_time_seconds < 120:
            return 0.3   # 有价值
        else:
            return 0.5   # 高价值
```

### 4.2 RL算法选择建议

#### 推荐算法：PPO (Proximal Policy Optimization)

```
┌─────────────────────────────────────────────────────────────────┐
│                    PPO适配架构                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────────────────────────────────────────────────┐ │
│   │                     Policy Network                        │ │
│   │   输入: 用户问题 + 意图嵌入 + 知识图谱状态                │ │
│   │   输出: 召回策略分布 (节点选择概率)                      │ │
│   └──────────────────────────────────────────────────────────┘ │
│                              │                                  │
│                              ▼                                  │
│   ┌──────────────────────────────────────────────────────────┐ │
│   │                     Value Network                         │ │
│   │   输入: 同上                                             │ │
│   │   输出: 状态价值估计 V(s)                                 │ │
│   └──────────────────────────────────────────────────────────┘ │
│                              │                                  │
│                              ▼                                  │
│   ┌──────────────────────────────────────────────────────────┐ │
│   │                     Reward Model                          │ │
│   │   从用户反馈学习，输出奖励信号                            │ │
│   └──────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 算法对比

| 算法 | 优点 | 缺点 | 推荐场景 |
|-----|------|------|---------|
| **PPO** | 稳定、样本效率高、超参数鲁棒 | 需要大量数据 | ✅ **推荐** |
| **DPO** | 无需奖励模型，直接偏好优化 | 需要成对偏好数据 | 有偏好数据集 |
| **REINFORCE** | 简单直观 | 方差大，不稳定 | 原型验证 |
| **A2C/A3C** | 并行采样高效 | 实现复杂 | 大规模部署 |

#### 为什么推荐PPO？

1. **稳定性好**：Clip机制防止策略崩溃
2. **样本效率高**：可重复使用样本
3. **适合在线学习**：增量更新友好
4. **工业界验证**：OpenAI、Anthropic广泛使用

#### PPO训练流程

```python
class PPOTrainer:
    def __init__(self, policy_network, value_network, clip_ratio=0.2):
        self.policy = policy_network
        self.value = value_network
        self.clip_ratio = clip_ratio
        
    def train_step(self, batch):
        """PPO训练步"""
        states, actions, rewards, next_states = batch
        
        # 计算优势函数
        values = self.value(states)
        next_values = self.value(next_states)
        advantages = rewards + 0.99 * next_values - values
        
        # 策略更新
        old_probs = self.policy(states).gather(actions)
        new_probs = self.policy(states).gather(actions)
        
        ratio = new_probs / old_probs
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        
        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()
        
        # 价值函数更新
        value_loss = F.mse_loss(self.value(states), rewards)
        
        return policy_loss, value_loss
```

### 4.3 完整RLHF流程

```
┌─────────────────────────────────────────────────────────────────┐
│                      RLHF完整流程                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Phase 1: 监督预训练 (SFT)                                     │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │   问题 → 模型 → 回答 → 专家标注 → 监督学习              │   │
│   │                                                         │   │
│   │   目标: 学习基础回答能力                                │   │
│   └─────────────────────────────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│   Phase 2: 奖励模型训练 (RM)                                    │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │   用户反馈 → 奖励函数 → 训练奖励模型                    │   │
│   │                                                         │   │
│   │   目标: 学习人类偏好                                    │   │
│   └─────────────────────────────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│   Phase 3: RL微调 (PPO)                                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │   问题 → Policy → 回答 → Reward Model → 分数 → PPO更新  │   │
│   │                                                         │   │
│   │   目标: 优化召回策略，最大化用户满意度                  │   │
│   └─────────────────────────────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│   Phase 4: 在线迭代                                            │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │   收集新反馈 → 更新奖励模型 → 继续PPO训练 → 循环        │   │
│   │                                                         │   │
│   │   目标: 持续适应用户偏好变化                            │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. 系统架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                        系统整体架构                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   用户问题                                                           │
│       │                                                             │
│       ▼                                                             │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    意图识别模块                              │   │
│   │   ┌─────────┐   ┌─────────────┐   ┌─────────────┐          │   │
│   │   │Embedding│──►│HDBSCAN聚类 │──►│意图类别分布 │          │   │
│   │   └─────────┘   └─────────────┘   └─────────────┘          │   │
│   │          │                                   │              │   │
│   │          ▼                                   ▼              │   │
│   │   ┌─────────────┐                    ┌─────────────┐       │   │
│   │   │实体识别增强 │                    │意图概率向量 │       │   │
│   │   └─────────────┘                    └─────────────┘       │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                           │                                         │
│                           ▼                                         │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    召回模块                                   │   │
│   │   ┌───────────────┐         ┌───────────────┐              │   │
│   │   │ 语义相似召回  │         │ 意图类别召回  │              │   │
│   │   │   (Path A)    │         │   (Path B)    │              │   │
│   │   └───────┬───────┘         └───────┬───────┘              │   │
│   │           │                         │                       │   │
│   │           └─────────┬───────────────┘                       │   │
│   │                     ▼                                        │   │
│   │           ┌─────────────────┐                               │   │
│   │           │   混合排序融合  │                               │   │
│   │           │   (Path C)      │                               │   │
│   │           └─────────────────┘                               │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                           │                                         │
│                           ▼                                         │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                  知识图谱检索                                 │   │
│   │   ┌─────────────────────────────────────────────────────┐  │   │
│   │   │  意图自适应遍历 → 候选节点 → 排序 → Top-K结果       │  │   │
│   │   └─────────────────────────────────────────────────────┘  │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                           │                                         │
│                           ▼                                         │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    回答生成                                   │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                           │                                         │
│                           ▼                                         │
│                      用户反馈                                       │
│                           │                                         │
│                           ▼                                         │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                   RLHF学习模块                                │   │
│   │   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │   │
│   │   │ 奖励函数    │──►│ 权重更新    │──►│ 策略优化    │       │   │
│   │   │ (显式+隐式) │   │ (动态权重)  │   │   (PPO)    │       │   │
│   │   └─────────────┘   └─────────────┘   └─────────────┘       │   │
│   │          │                                   │              │   │
│   │          └─────────────反馈循环──────────────┘              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. 实施建议

### 6.1 分阶段实施

| 阶段 | 内容 | 时间 | 优先级 |
|-----|------|-----|--------|
| **P0** | 基础语义召回 + 简单意图聚类 | 2周 | 高 |
| **P1** | 实体增强聚类 + 动态权重 | 3周 | 高 |
| **P2** | 双路召回 + 意图自适应 | 2周 | 中 |
| **P3** | RLHF完整流程 | 4周 | 中 |
| **P4** | 在线学习 + 持续优化 | 持续 | 低 |

### 6.2 关键指标

- **聚类质量**：轮廓系数 > 0.5，簇间距离 > 簇内距离
- **召回准确率**：Top-10召回准确率 > 80%
- **用户满意度**：正向反馈率 > 70%
- **响应时间**：端到端延迟 < 500ms

### 6.3 风险与缓解

| 风险 | 缓解措施 |
|-----|---------|
| 聚类不稳定 | 增量更新+周期重聚类 |
| 冷启动问题 | 预训练embedding + 规则兜底 |
| 过度拟合反馈 | 正则化 + 探索机制 |
| 数据稀疏 | 实体增强 + 知识图谱辅助 |

---

## 附录：核心数据结构

```python
@dataclass
class IntentCluster:
    """意图聚类"""
    cluster_id: str
    center_embedding: np.ndarray
    member_questions: List[str]
    representative_questions: List[str]  # 代表性问题
    entity_types: Dict[str, float]  # 主要实体类型分布
    created_at: datetime
    updated_at: datetime

@dataclass
class NodeWeight:
    """知识图谱节点权重"""
    node_id: str
    cluster_weights: Dict[str, float]  # {cluster_id: weight}
    feedback_count: int
    last_updated: datetime

@dataclass
class FeedbackEvent:
    """反馈事件"""
    user_id: str
    question: str
    cluster_id: str
    recalled_nodes: List[str]
    explicit_feedback: Dict[str, bool]  # {like, dislike, save, share}
    implicit_signals: Dict[str, float]  # {dwell_time, copy, requery}
    timestamp: datetime
```

---

*文档版本: v1.0 | 创建于 2026-03-25 | intent-clustering-team*