# 研究规划：多层级多维度意图召回系统

## 研究目标

构建一个对话式知识型 Agent 的意图分析系统，实现：
1. 多层级意图理解（表层→深层→元意图）
2. 多维度召回（语义+知识+上下文+用户模型）
3. 实时响应（<100ms）
4. 可解释性与可扩展性

---

## 一、研究阶段划分

### Phase 1: 基础验证（2-3个月）

**目标**: 验证核心技术可行性

#### 任务1.1: 意图层级定义与标注体系

```
输入: 现有意图数据集（MultiWOZ、SNIPS、ATIS）
输出: 
  - 三层意图层级定义文档
  - 中文意图标注规范
  - 试点标注数据集（500-1000对话）
```

**里程碑**:
- Week 1-2: 文献调研，定义层级结构
- Week 3-4: 标注规范制定
- Week 5-6: 试点标注与质量评估

#### 任务1.2: 单意图分类器基线

```
方法:
  - BERT-based classifier (基线)
  - Few-shot LLM (零样本对比)
  - Joint Intent-Slot model
  
评估指标:
  - Intent Accuracy
  - Slot F1
  - Joint Accuracy
  - Latency (<50ms)
```

**里程碑**:
- Week 1-2: 数据准备、模型训练
- Week 3: 评估与对比分析
- Week 4: 集成测试

#### 任务1.3: OOS检测初步方案

```
方法:
  - Threshold-based rejection
  - Uncertainty estimation (MC Dropout)
  - Prototype-based detection
  
验证:
  - 人工构造OOS样本
  - 检测率 vs 误拒率 曲线
```

---

### Phase 2: 多轮意图理解（3-4个月）

**目标**: 构建多轮对话中的意图理解能力

#### 任务2.1: 对话历史编码

```
架构选择:
  - Hierarchical encoder (utterance → dialogue)
  - Memory network
  - Transformer with position encoding
  
实验:
  - 不同编码方式的对比
  - 上下文窗口大小影响
```

#### 任务2.2: 意图状态追踪

```
方法:
  - Slot-level state tracking
  - Intent transition model
  - Goal state tracking
  
数据:
  - MultiWOZ (多领域)
  - Frames (目标演变)
  - 自建中文多轮数据集
```

#### 任务2.3: RAG增强的多轮意图

```
实现:
  - LARA方法复现
  - 历史相似对话检索
  - In-context examples生成
  
评估:
  - Zero-shot多轮意图准确率
  - 检索质量对准确率的影响
```

---

### Phase 3: 知识增强与推理（3-4个月）

**目标**: KG融合，实现推理层意图理解

#### 任务3.1: Intent Knowledge Graph构建

```
KG内容:
  - Intent节点（层级结构）
  - Intent-Intent关系（依赖、冲突、顺序）
  - Intent-Slot关系
  - Intent-Action关系
  
构建方法:
  - 专家定义 + 数据挖掘
  - Ontology设计
  - 增量更新机制
```

#### 任务3.2: KG-RAG融合架构

```
架构:
  ┌─────────────────────────────────────────────┐
  │  用户Query                                  │
  │      ↓                                      │
  │  实体识别 → KG子图提取                       │
  │      ↓                                      │
  │  向量检索 → Top-K文档                        │
  │      ↓                                      │
  │  KG路径 + 文档 → Prompt融合                  │
  │      ↓                                      │
  │  LLM推理 → 意图+理由                         │
  └─────────────────────────────────────────────┘
```

#### 任务3.3: 多意图推理

```
问题: 用户一句话包含多个意图
  
方法:
  - Multi-label classification (基线)
  - Intent Graph reasoning
  - Planning-based decomposition
  
创新点:
  - 意图依赖建模
  - 执行顺序推断
  - 冲突检测与解决
```

---

### Phase 4: 消歧与澄清（2-3个月）

**目标**: 构建智能消歧策略

#### 任务4.1: 消歧触发检测

```
触发条件:
  - Top-2 intent scores 接近
  - 置信度低于阈值
  - 槽位缺失导致无法执行
  
检测方法:
  - Entropy-based
  - Margin-based
  - Learned threshold
```

#### 任务4.2: Discriminative Questions生成

```
方法:
  - Rule-based: "您是想A还是想B？"
  - Model-based: Question generation
  - KG-based: 利用意图差异特征
  
评估:
  - 问题判别性（回答能否区分意图）
  - 用户理解度
  - 消歧成功率
```

#### 任务4.3: 消歧策略学习

```
方法:
  - Rule-based policy (基线)
  - RL-based clarification selection
  
奖励设计:
  - 消歧成功: +1
  - 减少对话轮数: +0.5
  - 用户满意度: 加权
```

---

### Phase 5: 系统集成与评估（3-4个月）

**目标**: 完整系统集成与全面评估

#### 任务5.1: 三层架构集成

```
Layer 1: 快速识别层 (<20ms)
  - 关键词匹配
  - 规则触发
  - BERT classifier
  
Layer 2: 语义理解层 (20-50ms)
  - 槽位填充
  - 上下文编码
  - 意图状态更新
  
Layer 3: 深度推理层 (50-100ms)
  - KG查询
  - LLM推理
  - 多意图分解
```

#### 任务5.2: 五维度评分融合

```
维度:
  1. Semantic similarity (向量匹配)
  2. Intent classifier score
  3. User profile match
  4. Context dependency
  5. Knowledge graph relevance
  
融合:
  - Weighted sum (可学习权重)
  - Attention-based fusion
  - Ranking model
```

#### 任务5.3: 端到端评估

```
评估维度:
  - Intent accuracy (离线)
  - Task success rate (在线)
  - Latency distribution
  - User satisfaction survey
  - A/B test with baseline
```

---

## 二、关键技术路径

### 路径A: 轻量级方案（适合快速部署）

```
┌─────────────────────────────────────────────────┐
│  MVP版本 (2周)                                   │
│  ───────────────────────────────────────────    │
│  - BERT intent classifier                       │
│  - Slot filling (CRF/BERT)                      │
│  - 规则-based消歧                               │
│  - Threshold-based OOS                          │
│                                                 │
│  增强版本 (4周)                                  │
│  ───────────────────────────────────────────    │
│  - +对话历史编码                                │
│  - +向量检索 (历史相似对话)                      │
│  - +用户画像                                    │
│                                                 │
│  完善版本 (6周)                                  │
│  ───────────────────────────────────────────    │
│  - +Intent KG                                   │
│  - +多意图检测                                  │
│  - +智能消歧                                    │
└─────────────────────────────────────────────────┘
```

### 路径B: 研究深入方案（适合学术研究）

```
Phase 1-2: 基础能力
  ───────────────────────
  - 多轮意图数据集构建
  - 层级意图建模论文
  - RAG增强方法论文
  
Phase 3-4: 创新突破
  ───────────────────────
  - Intent KG建模
  - 多意图推理算法
  - 消歧策略学习
  
Phase 5: 系统验证
  ───────────────────────
  - 端到端系统论文
  - 评估方法论贡献
```

---

## 三、数据需求与获取

### 公开数据集

| 数据集 | 用途 | 局限 |
|-------|------|------|
| **MultiWOZ** | 多领域任务导向对话 | 英文 |
| **SNIPS** | Intent+Slot评测 | 单轮 |
| **ATIS** | 航旅意图 | 单领域 |
| **Frames** | 目标演变对话 | 规模小 |

### 自建数据需求

| 数据类型 | 规模 | 方法 |
|---------|------|------|
| **中文多轮意图对话** | 5000+对话 | 人工标注+CLARA生成 |
| **Intent KG本体** | 50+意图类型 | 专家定义 |
| **消歧对话数据** | 1000+消歧轮 | 人工构造 |
| **用户画像数据** | 500+用户 | 模拟+真实 |

---

## 四、评估体系

### 离线评估

```
┌─────────────────────────────────────────────────┐
│  意图识别指标                                    │
│  ───────────────────────────────────────────    │
│  - Intent Accuracy (单意图)                     │
│  - Multi-Intent F1 (多意图)                     │
│  - Slot F1                                      │
│  - Joint Accuracy                               │
│  - OOS Detection Rate                           │
│                                                 │
│  多轮指标                                        │
│  ───────────────────────────────────────────    │
│  - Intent State Tracking Accuracy               │
│  - Goal Recognition Accuracy                    │
│  - Turn-level Accuracy                          │
│                                                 │
│  消歧指标                                        │
│  ───────────────────────────────────────────    │
│  - Disambiguation Success Rate                  │
│  - Clarifying Question Quality                  │
│  - Ambiguity Detection Accuracy                 │
└─────────────────────────────────────────────────┘
```

### 在线评估

```
┌─────────────────────────────────────────────────┐
│  任务指标                                        │
│  ───────────────────────────────────────────    │
│  - Task Success Rate                            │
│  - Turns to Complete                            │
│  - User Effort Score                            │
│                                                 │
│  性能指标                                        │
│  ───────────────────────────────────────────    │
│  - P50/P95/P99 Latency                          │
│  - Throughput                                   │
│  - Resource Usage                               │
│                                                 │
│  用户指标                                        │
│  ───────────────────────────────────────────    │
│  - Satisfaction Score (1-5)                     │
│  - Retention Rate                               │
│  - NPS (Net Promoter Score)                     │
└─────────────────────────────────────────────────┘
```

---

## 五、风险与应对

| 风险 | 概率 | 应对措施 |
|-----|------|---------|
| **语义落地问题无解** | 40% | KG锚定 + 交互式消歧 |
| **中文数据稀缺** | 70% | 跨语言迁移 + CLARA生成 |
| **多意图推理复杂度高** | 30% | 分层处理 + 启发式剪枝 |
| **实时性无法满足** | 20% | 缓存+量化+分级响应 |
| **用户画像冷启动** | 50% | 默认画像 + 快速学习 |

---

## 六、时间预算

| 方案 | 总时间 | 阶段划分 |
|-----|--------|---------|
| **轻量级方案** | 8-12周 | MVP 2周 + 增强 4周 + 完善 6周 |
| **研究深入方案** | 12-18月 | 5个Phase各3-4月 |
| **生产级系统** | 18-24月 | 研究12月 + 工程6月 + 优化6月 |

---

## 七、输出成果

### 研究输出

| 成果类型 | 内容 |
|---------|------|
| **论文** | 多轮意图理解、Intent KG建模、消歧策略学习 |
| **数据集** | 中文多轮意图对话、Intent KG本体 |
| **开源工具** | Intent classification toolkit、Disambiguation module |

### 系统输出

| 成果类型 | 内容 |
|---------|------|
| **架构设计** | 三层五维度意图召回系统 |
| **核心模块** | Intent classifier、State tracker、KG-RAG、Disambiguator |
| **评估框架** | 离线+在线评估工具 |

---

**规划文档生成时间**: 2026-03-29
**项目目录**: ~/clawteam-projects/intent-world-projection