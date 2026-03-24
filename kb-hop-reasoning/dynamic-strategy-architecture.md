# 动态意图策略权重架构设计

**生成时间**: 2026-03-24  
**类型**: 架构设计（基于用户深度洞察）  
**重要性**: ⭐⭐⭐⭐⭐ 核心实现方案

---

## 一、设计理念

### 1.1 核心思想

> **以语义向量为核心，意图策略驱动，动态扩展权重项，按策略召回排序**

### 1.2 与传统方案对比

| 维度 | 传统方案 | 本方案 |
|------|----------|--------|
| 权重表示 | 标量/固定向量 | 语义向量 + 动态权重项 |
| 权重粒度 | 固定聚类 | 意图策略（可动态扩展） |
| 召回策略 | 单一排序 | 多策略自适应 |
| 扩展性 | 需重新设计 | 新增策略即可 |
| 可解释性 | 中等 | 高（策略对应意图） |

### 1.3 设计优势

```
✅ 语义优先：语义向量作为核心表示
✅ 策略驱动：不同意图使用不同权重策略
✅ 动态扩展：新意图策略可无缝添加
✅ 可组合性：策略可组合、权重可叠加
✅ 端到端：意图识别 → 策略选择 → 召回排序 一体化
```

---

## 二、架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        动态意图策略权重系统                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ 用户问题    │───→│ 意图识别    │───→│ 策略匹配    │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                            │                │                   │
│                            ↓                ↓                   │
│                     ┌─────────────────────────────┐            │
│                     │      意图策略池             │            │
│                     │  ┌───────┐ ┌───────┐      │            │
│                     │  │策略1  │ │策略2  │ ...  │            │
│                     │  └───────┘ └───────┘      │            │
│                     └─────────────────────────────┘            │
│                                │                                │
│                                ↓                                │
│                     ┌─────────────────────────────┐            │
│                     │      权重项计算             │            │
│                     │  [项1, 项2, 项3, ...]       │            │
│                     └─────────────────────────────┘            │
│                                │                                │
│                                ↓                                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │候选元素     │───→│ 权重打分    │───→│ 排序召回    │         │
│  │(实体/关系)  │    │             │    │             │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件

#### 组件1：语义向量层

```
实体语义向量:
  entity.embedding = EmbeddingModel.encode(entity.text)
  维度: 768 (BERT-base) 或 1536 (OpenAI)

关系语义向量:
  relation.embedding = EmbeddingModel.encode(relation.text)
  
路径语义向量:
  path.embedding = aggregate([entity.embeddings, relation.embeddings])
```

#### 组件2：意图策略池

```python
class IntentStrategy:
    """意图策略定义"""
    
    strategy_id: str           # 策略ID
    strategy_name: str         # 策略名称
    intent_keywords: List[str] # 触发关键词
    intent_embedding: np.array # 意图向量
    weight_items: List[str]    # 权重项列表
    weight_vector: np.array    # 权重向量
    aggregation_func: str      # 聚合函数
```

#### 组件3：权重项计算器

```python
class WeightItemCalculator:
    """权重项计算器"""
    
    def compute_reliability(self, entity, context):
        """可靠性"""
        return self._get_source_reliability(entity.source)
    
    def compute_temporal_relevance(self, entity, context):
        """时效相关性"""
        return self._check_freshness(entity.last_updated, context.time)
    
    def compute_association_strength(self, relation, context):
        """关联强度"""
        return relation.frequency / self._total_frequency
    
    def compute_evidence_sufficiency(self, path, context):
        """证据充分性"""
        return self._check_path_completeness(path)
    
    # ... 更多权重项
```

#### 组件4：策略路由器

```python
class StrategyRouter:
    """策略路由器 - 根据意图选择策略"""
    
    def select_strategy(self, intent_vector):
        """
        选择最匹配的策略
        
        可以是:
        1. 单策略选择 (argmax)
        2. 多策略组合 (weighted sum)
        3. 层级策略 (先粗后细)
        """
        scores = {}
        for strategy in self.strategies:
            score = cosine_similarity(intent_vector, strategy.intent_embedding)
            scores[strategy.strategy_id] = score
        
        return self._select_top_k(scores, k=1)
```

---

## 三、意图策略定义

### 3.1 预定义策略

#### 策略1：事实查询 (Fact Query)

```yaml
strategy_id: fact_query
strategy_name: 事实查询
intent_keywords: [是什么, 多少, 哪里, 什么时候, 谁]
weight_items:
  - reliability: 0.35        # 数据来源可靠性
  - temporal_relevance: 0.25 # 时效性
  - source_authority: 0.25   # 来源权威度
  - semantic_match: 0.15     # 语义匹配度
aggregation: weighted_sum
```

#### 策略2：关系推理 (Relation Inference)

```yaml
strategy_id: relation_inference
strategy_name: 关系推理
intent_keywords: [的关系, 怎么联系, 是否关联]
weight_items:
  - association_strength: 0.30  # 关联强度
  - path_credibility: 0.30      # 路径可信度
  - evidence_sufficiency: 0.25  # 证据充分性
  - semantic_match: 0.15        # 语义匹配度
aggregation: weighted_sum
```

#### 策略3：多跳推理 (Multi-hop Reasoning)

```yaml
strategy_id: multi_hop_reasoning
strategy_name: 多跳推理
intent_keywords: [的上级的, 的领导的, 的...的]  # 连锁结构
weight_items:
  - path_coherence: 0.30       # 路径连贯性
  - hop_confidence: 0.25       # 每跳置信度衰减
  - evidence_chain: 0.25       # 证据链完整度
  - semantic_alignment: 0.20   # 语义对齐度
aggregation: multiplicative    # 乘法聚合 (强调短板)
```

#### 策略4：时效敏感 (Temporal Sensitive)

```yaml
strategy_id: temporal_sensitive
strategy_name: 时效敏感
intent_keywords: [最新, 当前, 现在, 最近]
weight_items:
  - temporal_relevance: 0.40   # 时效性
  - update_frequency: 0.30     # 更新频率
  - expiration_risk: 0.20      # 过期风险
  - semantic_match: 0.10       # 语义匹配度
aggregation: weighted_sum
```

#### 策略5：因果解释 (Causal Explanation)

```yaml
strategy_id: causal_explanation
strategy_name: 因果解释
intent_keywords: [为什么, 原因, 导致, 影响]
weight_items:
  - causal_strength: 0.35      # 因果强度
  - evidence_chain: 0.30       # 证据链
  - plausibility: 0.20         # 合理性
  - semantic_match: 0.15       # 语义匹配度
aggregation: weighted_sum
```

#### 策略6：比较分析 (Comparative Analysis)

```yaml
strategy_id: comparative_analysis
strategy_name: 比较分析
intent_keywords: [比, 对比, 差异, 相同]
weight_items:
  - comparability: 0.30        # 可比性
  - attribute_completeness: 0.30  # 属性完整度
  - data_accuracy: 0.25        # 数据准确性
  - semantic_match: 0.15       # 语义匹配度
aggregation: weighted_sum
```

### 3.2 策略扩展机制

```python
def register_new_strategy(strategy_config):
    """
    注册新策略
    
    支持运行时动态扩展，无需重启系统
    """
    strategy = IntentStrategy(
        strategy_id=strategy_config["strategy_id"],
        strategy_name=strategy_config["strategy_name"],
        intent_keywords=strategy_config["intent_keywords"],
        intent_embedding=encode_keywords(strategy_config["intent_keywords"]),
        weight_items=strategy_config["weight_items"],
        weight_vector=normalize(strategy_config["weight_vector"]),
        aggregation_func=strategy_config["aggregation"]
    )
    
    strategy_pool.register(strategy)
```

---

## 四、权重计算流程

### 4.1 完整流程

```python
def dynamic_weight_retrieval(question: str, knowledge_graph: Graph) -> List[Result]:
    """
    动态意图策略权重召回
    
    Args:
        question: 用户问题
        knowledge_graph: 知识图谱
    
    Returns:
        召回结果列表
    """
    
    # ========== Step 1: 意图识别 ==========
    intent_vector = intent_classifier.encode(question)
    # 例: "姚明妻子的出生地？" → [0.1, 0.2, 0.7, 0.1, ...]
    #                     偏向"多跳推理"意图
    
    # ========== Step 2: 策略选择 ==========
    matched_strategies = strategy_router.select_top_k(intent_vector, k=2)
    # 例: 返回 [("multi_hop_reasoning", 0.85), ("fact_query", 0.30)]
    
    # ========== Step 3: 候选生成 ==========
    # 从图谱中检索候选实体/关系/路径
    candidates = candidate_generator.generate(question, knowledge_graph)
    
    # ========== Step 4: 权重计算 ==========
    scored_candidates = []
    
    for candidate in candidates:
        # 获取语义向量
        semantic_vec = candidate.get_semantic_embedding()
        
        # 计算各策略下的分数
        strategy_scores = {}
        
        for strategy, match_score in matched_strategies:
            # 获取该策略的权重项
            weight_items = strategy.weight_items
            
            # 计算各项权重分数
            item_scores = {}
            for item_name, item_weight in weight_items.items():
                item_scores[item_name] = weight_calculator.compute(
                    item_name, candidate, question
                )
            
            # 聚合得到策略分数
            strategy_score = aggregate(
                item_scores, 
                strategy.weight_vector,
                strategy.aggregation_func
            )
            
            strategy_scores[strategy.strategy_id] = strategy_score
        
        # 综合分数 = 意图匹配度 × 策略分数
        final_score = sum(
            match_score * strategy_scores[strategy_id]
            for strategy_id, match_score in matched_strategies
        )
        
        # 语义相似度加成
        semantic_bonus = cosine_similarity(semantic_vec, intent_vector)
        final_score = final_score * (1 + 0.2 * semantic_bonus)
        
        scored_candidates.append((candidate, final_score))
    
    # ========== Step 5: 排序召回 ==========
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    
    return scored_candidates[:TOP_K]
```

### 4.2 多跳推理示例

```
问题: "姚明妻子的出生地？"

Step 1: 意图识别
  intent_vector = [0.1, 0.2, 0.7, 0.1, ...]
  主意图: multi_hop_reasoning (0.85)

Step 2: 策略选择
  主策略: multi_hop_reasoning
  权重项: [path_coherence, hop_confidence, evidence_chain, semantic_alignment]

Step 3: 候选路径
  路径A: 姚明 → 妻子 → 叶莉 → 出生地 → 上海
  路径B: 姚明 → 配偶 → 叶莉 → 籍贯 → 上海
  路径C: 姚明 → 妻子 → 叶莉 → 家乡 → 上海

Step 4: 权重计算 (路径A)
  path_coherence = 0.90      (关系连贯)
  hop_confidence = 0.85      (每跳可信)
  evidence_chain = 0.95      (证据链完整)
  semantic_alignment = 0.88  (语义对齐)
  
  aggregation = 0.90 × 0.85 × 0.95 × 0.88 = 0.64 (乘法聚合)

Step 5: 最终排序
  路径A: 0.64 → 排名1 → 召回
  路径B: 0.58 → 排名2
  路径C: 0.52 → 排名3
```

---

## 五、聚合策略

### 5.1 聚合函数类型

```python
def aggregate(item_scores, weight_vector, aggregation_func):
    """
    聚合权重项分数
    """
    
    if aggregation_func == "weighted_sum":
        # 加权求和 (适用于独立权重项)
        return sum(s * w for s, w in zip(item_scores.values(), weight_vector))
    
    elif aggregation_func == "multiplicative":
        # 乘法聚合 (强调短板，适用于链式依赖)
        return product(item_scores.values()) ** (1 / len(item_scores))
    
    elif aggregation_func == "max":
        # 取最大值 (适用于可选权重项)
        return max(item_scores.values())
    
    elif aggregation_func == "min":
        # 取最小值 (适用于安全关键场景)
        return min(item_scores.values())
    
    elif aggregation_func == "harmonic_mean":
        # 调和平均 (平衡各项，惩罚极端低值)
        return len(item_scores) / sum(1/s for s in item_scores.values())
```

### 5.2 策略-聚合函数映射

| 策略类型 | 推荐聚合函数 | 原因 |
|----------|-------------|------|
| 事实查询 | weighted_sum | 各维度独立 |
| 关系推理 | weighted_sum | 各维度独立 |
| 多跳推理 | multiplicative | 链式依赖，强调短板 |
| 时效敏感 | harmonic_mean | 平衡时效与其他维度 |
| 因果解释 | weighted_sum | 各维度独立 |
| 比较分析 | min | 确保可比性 |

---

## 六、策略组合

### 6.1 多策略组合场景

```
场景: 用户问"姚明妻子最近的工作是什么？"

意图分析:
  1. 多跳推理 (姚明 → 妻子 → 叶莉 → 工作)
  2. 时效敏感 (最近的工作)

策略组合:
  multi_hop_reasoning: 0.60
  temporal_sensitive: 0.40

组合权重计算:
  final_score = 0.60 * score_multi_hop + 0.40 * score_temporal
```

### 6.2 策略组合实现

```python
def combine_strategies(strategy_scores, intent_weights):
    """
    组合多策略分数
    
    Args:
        strategy_scores: {strategy_id: score}
        intent_weights: {strategy_id: weight}
    
    Returns:
        combined_score
    """
    combined = sum(
        strategy_scores[sid] * intent_weights.get(sid, 0)
        for sid in strategy_scores
    )
    
    # 归一化
    total_weight = sum(intent_weights.values())
    return combined / total_weight if total_weight > 0 else 0
```

---

## 七、动态扩展机制

### 7.1 运行时策略注册

```python
# 新增"推荐"策略
new_strategy = {
    "strategy_id": "recommendation",
    "strategy_name": "推荐",
    "intent_keywords": ["推荐", "建议", "适合"],
    "weight_items": {
        "personalization": 0.35,
        "popularity": 0.25,
        "relevance": 0.25,
        "diversity": 0.15
    },
    "aggregation": "weighted_sum"
}

# 注册到策略池
strategy_pool.register(new_strategy)

# 立即生效，无需重启
```

### 7.2 权重项动态注册

```python
# 新增"个性化"权重项
@weight_calculator.register("personalization")
def compute_personalization(entity, context):
    """
    个性化权重项
    """
    user_profile = context.user_profile
    entity_features = entity.features
    
    # 计算用户偏好与实体特征的匹配度
    return cosine_similarity(user_profile, entity_features)
```

### 7.3 A/B测试支持

```python
# 策略A/B测试
experiment_config = {
    "experiment_id": "exp_001",
    "strategies": {
        "control": ["fact_query"],
        "treatment": ["fact_query_v2"]  # 新版策略
    },
    "traffic_split": {"control": 0.5, "treatment": 0.5}
}

# 运行实验
result = experiment_runner.run(experiment_config)
```

---

## 八、实现架构

### 8.1 技术栈

```
┌─────────────────────────────────────────────────────────────┐
│                      应用层                                  │
│  API Gateway / gRPC Service                                 │
├─────────────────────────────────────────────────────────────┤
│                      服务层                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │意图识别服务 │  │策略路由服务 │  │权重计算服务 │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│                      数据层                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Neo4j      │  │ Milvus     │  │ Redis      │         │
│  │ (图谱存储)  │  │ (向量存储)  │  │ (策略缓存)  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│                      基础设施                                │
│  Kubernetes / Docker / 监控告警                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 数据结构

```python
# 策略配置表 (存储在 PostgreSQL / MongoDB)
{
    "strategy_id": "multi_hop_reasoning",
    "strategy_name": "多跳推理",
    "intent_keywords": ["的上级的", "的领导的"],
    "intent_embedding": [0.1, 0.2, ...],  # 预计算
    "weight_items": {
        "path_coherence": 0.30,
        "hop_confidence": 0.25,
        "evidence_chain": 0.25,
        "semantic_alignment": 0.20
    },
    "aggregation_func": "multiplicative",
    "created_at": "2026-03-24T00:00:00Z",
    "updated_at": "2026-03-24T00:00:00Z",
    "status": "active"
}

# 权重项定义表
{
    "item_name": "path_coherence",
    "item_description": "路径连贯性",
    "compute_function": "compute_path_coherence",
    "value_range": [0, 1],
    "dependencies": ["path"]
}

# 召回日志表 (用于策略优化)
{
    "query_id": "q_001",
    "question": "姚明妻子的出生地？",
    "intent_vector": [0.1, 0.2, 0.7, ...],
    "matched_strategies": ["multi_hop_reasoning"],
    "candidates": [
        {"path": "...", "score": 0.64, "rank": 1}
    ],
    "final_answer": "上海",
    "user_feedback": "correct",
    "timestamp": "2026-03-24T11:30:00Z"
}
```

---

## 九、监控与优化

### 9.1 监控指标

```yaml
策略层面:
  - 各策略使用频率
  - 各策略准确率
  - 策略响应时间

权重项层面:
  - 各权重项分布
  - 权重项区分度
  - 权重项稳定性

系统层面:
  - 召回延迟 P50/P95/P99
  - 策略匹配准确率
  - 用户满意度
```

### 9.2 优化闭环

```
┌─────────────────────────────────────────────────────────────┐
│                     优化闭环                                 │
│                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │ 数据收集│───→│ 离线分析│───→│ 策略优化│───→│ 在线更新│  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
│       ↑                                              │      │
│       └──────────────────────────────────────────────┘      │
│                                                             │
│  优化动作:                                                  │
│  1. 调整权重项权重                                          │
│  2. 新增/下线策略                                           │
│  3. 更新意图关键词                                          │
│  4. 调整聚合函数                                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 十、总结

### 10.1 核心设计原则

| 原则 | 说明 |
|------|------|
| **语义优先** | 语义向量作为核心表示 |
| **策略驱动** | 意图决定策略，策略决定权重 |
| **动态扩展** | 新策略/权重项可运行时注册 |
| **可组合** | 多策略组合，权重可叠加 |
| **可观测** | 完善的监控和优化闭环 |

### 10.2 与前方案对比

| 维度 | 语义聚类权重方案 | 动态意图策略方案 |
|------|-----------------|-----------------|
| 权重粒度 | 聚类级别 | 意图策略级别 |
| 灵活性 | 中等 | 高 |
| 扩展性 | 需重新设计聚类 | 运行时注册 |
| 语义对齐 | ✅ | ✅ |
| 动态性 | 批量更新 | 实时调整 |
| 实现复杂度 | 中等 | 较高 |

### 10.3 推荐实施路径

```
Phase 1: 基础框架 (2周)
  ├─ 意图识别模型
  ├─ 策略路由器
  └─ 基础权重计算

Phase 2: 策略定义 (1周)
  ├─ 预定义6个核心策略
  └─ 权重项计算器

Phase 3: 动态扩展 (1周)
  ├─ 运行时策略注册
  ├─ 权重项注册
  └─ A/B测试框架

Phase 4: 优化闭环 (持续)
  ├─ 监控指标
  ├─ 离线分析
  └─ 自动优化
```

---

*设计者: Nemo (基于用户深度洞察)*  
*生成时间: 2026-03-24*  
*重要性: 核心实现架构*