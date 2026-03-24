# SOTA 知识图谱多跳推理方法调研报告

**调研时间**: 2026-03-24  
**调研范围**: 2024-2025年最新研究成果  
**技术路线**: GNN-based、Path-based、LLM-enhanced

---

## 一、概述

知识图谱多跳推理（Multi-hop Reasoning on Knowledge Graphs）是指通过在知识图谱上进行多步推理来回答复杂问题的技术。近年来，随着大语言模型的发展，该领域涌现出多种融合方案。本报告按照三大主流技术路线进行分类整理。

---

## 二、LLM-Enhanced 方法

### 2.1 GraphRAG 系列方法

#### 1. BubbleRAG: Evidence-Driven Retrieval-Augmented Generation for Black-Box Knowledge Graphs
- **论文**: arXiv:2603.20309 (2026.03)
- **核心思想**: 针对黑盒知识图谱的 RAG 问题，提出 Optimal Informative Subgraph Retrieval (OISR) 问题形式化
- **关键技术**:
  - Semantic anchor grouping（语义锚点分组）
  - Heuristic bubble expansion（启发式气泡扩展）
  - Composite ranking + reasoning-aware expansion
- **贡献**: 在多跳 QA 基准上达到 SOTA，无需训练即可使用

#### 2. A Robust Multi-Hop GraphRAG Retrieval Framework (C2RAG)
- **论文**: arXiv:2603.14828 (2026.03)
- **核心思想**: 解决知识图谱质量问题导致的检索漂移和检索幻觉
- **关键技术**:
  - Constraint-based retrieval（约束检索）：将查询分解为原子约束三元组
  - Sufficiency check（充分性检查）：判断当前证据是否足以支持推理
- **性能**: 平均提升 3.4% EM 和 3.9% F1

#### 3. The Reasoning Bottleneck in Graph-RAG
- **论文**: arXiv:2603.14045 (2026.03)
- **核心思想**: 发现 Graph-RAG 系统中 73-84% 的错误是推理失败而非检索失败
- **关键技术**:
  - SPARQL chain-of-thought prompting
  - Graph-walk compression（压缩约 60%）
- **亮点**: Llama-8B + 增强方法可匹配或超越 Llama-70B，成本降低 12 倍

#### 4. From Flat to Structural: Enhancing Automated Short Answer Grading with GraphRAG
- **论文**: arXiv:2603.19276 (2026.02)
- **核心思想**: 将"扁平"向量检索升级为结构化知识图谱
- **关键技术**:
  - Microsoft GraphRAG 用于图谱构建
  - HippoRAG neurosymbolic 算法执行关联图遍历
- **应用**: 教育评估领域

### 2.2 Agent-based 方法

#### 5. Explore-on-Graph (EoG)
- **论文**: arXiv:2602.21728 (2026.02)
- **核心思想**: 通过强化学习激励 LLM 自主探索知识图谱
- **关键技术**:
  - Path-refined reward modeling
  - 基于推理路径正确性的奖励信号
- **优势**: 突破固定示范路径的限制，发现新颖推理路径

#### 6. RouterKGQA: Specialized-General Model Routing
- **论文**: arXiv:2603.20017 (2026.03)
- **核心思想**: 专用模型 + 通用模型协作路由
- **关键技术**:
  - 专用模型生成推理路径
  - 通用模型进行 KG-guided 修复
  - Constraint-aware answer filtering
- **性能**: F1 提升 3.57，平均仅需 1.15 次 LLM 调用

#### 7. D-MEM: Dopamine-Gated Agentic Memory
- **论文**: arXiv:2603.14597 (2026.03)
- **核心思想**: 受多巴胺启发的快/慢路由系统
- **关键技术**:
  - Reward Prediction Error (RPE) 路由
  - O(1) 快速访问缓冲区 + O(N) 记忆演化管道
- **性能**: 减少 80%+ token 消耗，消除 O(N²) 瓶颈

#### 8. RareAgent: Self-Evolving Reasoning for Drug Repurposing
- **论文**: arXiv:2510.05764 (2025.10)
- **核心思想**: 多智能体对抗辩论构建证据图
- **关键技术**:
  - 自演化循环：后验分析 + 策略精化
  - 可迁移启发式蒸馏
- **性能**: indication AUPRC 提升 18.1%

---

## 三、GNN-Based 方法

### 3.1 神经符号融合

#### 9. HYQNET: Neural-Symbolic Logic Query Answering in Non-Euclidean Space
- **论文**: arXiv:2603.15633 (2026.02)
- **核心思想**: 在双曲空间中进行逻辑查询推理
- **关键技术**:
  - FOL 查询分解为关系投影和逻辑操作
  - 双曲 GNN 进行知识图谱补全
- **优势**: 更好地捕捉逻辑投影推理的层次结构

#### 10. PN-GNN: Enhancing Logical Expressiveness via Path-Neighbor Aggregation
- **论文**: arXiv:2511.07994 (AAAI 2026)
- **核心思想**: 通过在推理路径上聚合节点邻居嵌入增强逻辑表达能力
- **关键技术**:
  - Path-Neighbor 聚合机制
  - 理论证明: (k+1)-hop 表达能力严格优于 k-hop
- **贡献**: 增强逻辑规则表达能力而不牺牲泛化性

### 3.2 时序知识图谱推理

#### 11. IGETR: Integration of Graph and Editing-enhanced Temporal Reasoning
- **论文**: arXiv:2601.21978 (2026.01)
- **核心思想**: GNN 结构化建模 + LLM 上下文理解的混合框架
- **关键技术**:
  - Temporal GNN 识别时空一致的候选路径
  - LLM-guided path editing 修复语义不一致
  - 路径集成预测
- **性能**: ICEWS 数据集 Hits@1 提升 5.6%，Hits@3 提升 8.1%

---

## 四、Path-Based 方法

### 4.1 规则引导路径探索

#### 12. TRACE: Temporal Rule-Anchored Chain-of-Evidence
- **论文**: arXiv:2603.12500 (2026.03)
- **核心思想**: 规则引导的多跳探索 + 新闻锚定 + LLM 决策
- **关键技术**:
  - 规则引导探索（限制在可接受的关系序列）
  - 候选推理链锚定新闻
  - 可审计的 UP/DOWN 判决
- **应用**: 股票预测（S&P 500）
- **性能**: F1 60.8%，召回率 71.5%

---

## 五、技术路线对比

| 技术路线 | 优势 | 劣势 | 适用场景 |
|---------|------|------|---------|
| **LLM-Enhanced** | 语义理解强、泛化好、可解释性佳 | 计算成本高、依赖 LLM 质量 | 复杂问答、开放域推理 |
| **GNN-Based** | 结构信息利用充分、推理效率高 | 图谱不完整时性能下降 | 已知图谱上的推理任务 |
| **Path-Based** | 推理路径可解释、规则可迁移 | 路径搜索空间大、规则获取困难 | 特定领域推理、可审计场景 |

---

## 六、发展趋势

1. **LLM + GNN 融合**: 利用 GNN 捕捉结构信息，LLM 处理语义推理
2. **神经符号系统**: 结合符号推理的可解释性和神经网络的泛化能力
3. **自适应检索**: 根据问题复杂度动态选择检索深度和模型规模
4. **强化学习引导**: 通过奖励信号优化推理路径探索
5. **双曲空间表示**: 更好地捕捉层次结构和逻辑关系

---

## 七、代表性论文列表

| 序号 | 论文标题 | arXiv ID | 技术路线 | 发表时间 |
|-----|---------|----------|---------|---------|
| 1 | BubbleRAG | 2603.20309 | LLM-Enhanced | 2026.03 |
| 2 | C2RAG | 2603.14828 | LLM-Enhanced | 2026.03 |
| 3 | Reasoning Bottleneck in Graph-RAG | 2603.14045 | LLM-Enhanced | 2026.03 |
| 4 | GraphRAG for ASAG | 2603.19276 | LLM-Enhanced | 2026.02 |
| 5 | Explore-on-Graph (EoG) | 2602.21728 | LLM-Enhanced | 2026.02 |
| 6 | RouterKGQA | 2603.20017 | LLM-Enhanced | 2026.03 |
| 7 | D-MEM | 2603.14597 | LLM-Enhanced | 2026.03 |
| 8 | RareAgent | 2510.05764 | LLM-Enhanced | 2025.10 |
| 9 | HYQNET | 2603.15633 | GNN-Based | 2026.02 |
| 10 | PN-GNN | 2511.07994 | GNN-Based | AAAI 2026 |
| 11 | IGETR | 2601.21978 | GNN+LLM | 2026.01 |
| 12 | TRACE | 2603.12500 | Path-Based | 2026.03 |

---

## 八、常用基准数据集

- **HotpotQA**: 多跳问答基准
- **MuSiQue**: 多跳问答数据集
- **2WikiMultiHopQA**: Wikipedia 多跳推理
- **ICEWS**: 时序知识图谱事件预测
- **MetaQA**: 电影领域多跳问答

---

**调研者**: worker1 (ClawTeam kb-hop-reasoning)  
**报告版本**: v1.0