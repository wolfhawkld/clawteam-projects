# Mamba/SSM 用于 Agent Memory 架构研究调研

> 调研时间：2026-04-12
> 调研目的：判断是否存在将 Mamba/SSM 的选择性机制应用于 Agent 记忆架构的相关研究

---

## 1. 调研结论

**核心发现：目前没有直接将 Mamba/SSM 的选择性机制用于 Agent Memory 架构的研究。这是一个空白领域。**

---

## 2. Agent Memory 架构研究现状

### 2.1 主要论文

| 论文 | 年份 | 引用 | 核心方法 |
|------|------|------|---------|
| **Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory** | 2025 | 12 | 生产级长期记忆系统，向量存储+检索 |
| **KARMA: Augmenting Embodied AI Agents with Long-and-short Term Memory Systems** | 2025 | 5 | 长短记忆分离，类人类记忆结构 |
| **From LLM to Conversational Agent: A Memory Enhanced Architecture** | 2024 | 6 | 记忆增强架构，Fine-tuning LLM |
| **Memory Sandbox: Transparent and Interactive Memory Management** | 2023 | 27 | 透明交互式记忆管理，用户可控 |
| **Human-Like Remembering and Forgetting in LLM Agents: An ACT-R-Inspired Memory Architecture** | 2025 | 1 | ACT-R 认知架构启发，模拟人类遗忘 |

### 2.2 现有方法的特点

| 方法 | 记忆写入策略 | 记忆压缩 | 选择性机制 |
|------|------------|---------|-----------|
| Mem0 | 全量写入 | 无压缩 | 检索时过滤（向量相似度） |
| KARMA | 分层写入（长/短） | 时间衰减 | 无语义选择性 |
| Memory Enhanced Architecture | 规则写入 | 无 | 无 |
| ACT-R Inspired | 模拟遗忘曲线 | 时间衰减 | 基于认知理论，非学习得到 |
| Memory Sandbox | 用户控制 | 手动 | 用户交互选择 |

**共同问题**：记忆写入时缺乏"语义选择性"——写入什么、压缩什么，主要靠规则或时间衰减，而非根据内容语义动态决策。

---

## 3. Mamba/SSM 相关研究现状

### 3.1 Mamba 应用领域

| 领域 | 代表论文 | 引用 | 说明 |
|------|---------|------|------|
| 视觉 | Vision Mamba (2024) | 386 | 双向 SSM 用于图像 |
| 医学图像 | SegMamba (2024) | 392 | 3D 医学图像分割 |
| 多模态 | Cobra (2025) | 32 | 多模态 LLM |
| 混合架构 | Jamba (2024) | 40 | Transformer + Mamba 混合 |
| 导航 | Memory-MambaNav (2025) | 新 | 物体导航，用 Mamba 处理空间记忆 |

### 3.2 Memory-MambaNav 唯一相关研究

**Memory-MambaNav: Enhancing object-goal navigation through memory-based Mamba architecture** (2025)

- 用 Mamba 处理**空间记忆**（导航轨迹）
- 不是对话/语义记忆，而是物理空间记忆
- 说明 Mamba 在"记忆"场景有潜力，但尚未扩展到 Agent 对话记忆

---

## 4. Memory Compression 相关研究

| 论文 | 年份 | 核心方法 | 与 SSM 关系 |
|------|------|---------|------------|
| Dynamic Memory Compression: Retrofitting LLMs for Accelerated Inference | 2024 | KV Cache 动态压缩 | 无，基于注意力 |
| Ecco: Improving Memory Bandwidth via Entropy-Aware Cache Compression | 2025 | 熵感知压缩 | 无 |

**问题**：现有 Memory Compression 基于 Transformer/KV Cache，未利用 SSM 的线性压缩优势。

---

## 5. 研究空白分析

### 5.1 空白领域

| 可能方向 | 现状 | 机会 |
|---------|------|------|
| **选择性记忆写入** | 无 | 用 Mamba 的 B_t 思想，设计"语义→写入强度"评估器 |
| **动态记忆压缩** | 时间衰减为主 | 用 SSM 的 Δ_t 思想，设计"遗忘速率"学习机制 |
| **选择性记忆检索** | 向量相似度 | 用 Mamba 的 C_t 思想，设计"意图→检索焦点"机制 |
| **固定状态记忆** | 向量库（无限增长） | 用 SSM 固定隐藏状态，实现 bounded memory |

### 5.2 Mamba 思想的可迁移性

| Mamba 机制 | Agent Memory 对应 | 研究价值 |
|-----------|------------------|---------|
| B_t (选择性写入) | 语义重要性评估 → 写入决策 | **高**，尚未有人做 |
| Δ_t (遗忘节奏) | 记忆衰减速率学习 | **高**，现有方法是固定衰减 |
| C_t (选择性输出) | 检索聚焦机制 | **中**，现有检索是静态的 |
| 固定状态 h_t | bounded memory | **高**，现有是无限增长向量库 |

---

## 6. 结论与建议

### 6.1 结论

**这是一个有价值的研究空白：**

1. **Agent Memory 架构**已有研究，但缺乏"语义选择性"机制
2. **Mamba/SSM**的选择性机制尚未迁移到 Agent Memory
3. **Memory Compression**存在，但未与 SSM 结合
4. **唯一相关研究**（Memory-MambaNav）针对空间记忆，非对话语义记忆

### 6.2 建议研究方向

1. **Selective Memory Write**：借鉴 Mamba B_t，设计"语义→写入决策"的可学习评估器
2. **Dynamic Memory Decay**：借鉴 Mamba Δ_t，设计可学习的记忆衰减机制
3. **Bounded Memory Architecture**：借鉴 SSM 固定状态，设计 bounded 大小的记忆系统
4. **Intent-aware Retrieval**：借鉴 Mamba C_t，设计意图驱动的选择性检索

### 6.3 潜在挑战

| 挑战 | 说明 |
|------|------|
| 训练数据 | 如何获得"重要性"标签？可能需要用户反馈或任务成败信号 |
| 评估指标 | 如何评估"记忆质量"？检索准确率、任务完成度？ |
| 与现有系统集成 | 如何与 RAG、Vector DB 等现有技术结合？ |

---

## 7. 参考文献

### Agent Memory

1. Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory - https://arxiv.org/abs/2504.19413
2. KARMA: Augmenting Embodied AI Agents with Long-and-short Term Memory Systems - https://arxiv.org/abs/2409.14908
3. From LLM to Conversational Agent: A Memory Enhanced Architecture - https://arxiv.org/abs/2401.02777
4. Memory Sandbox: Transparent and Interactive Memory Management - https://doi.org/10.1145/3586182.3615796
5. Human-Like Remembering and Forgetting in LLM Agents: An ACT-R-Inspired Memory Architecture - https://doi.org/10.1145/3765766.3765803

### Mamba/SSM

1. Mamba: Linear-Time Sequence Modeling with Selective State Spaces - https://arxiv.org/abs/2312.00752
2. Vision Mamba - https://arxiv.org/abs/2401.09417
3. Jamba: Hybrid Transformer-Mamba Language Model - https://arxiv.org/abs/2403.19887
4. Memory-MambaNav - https://www.sciencedirect.com/science/article/abs/pii/S0262885625001106

### Memory Compression

1. Dynamic Memory Compression: Retrofitting LLMs for Accelerated Inference - https://arxiv.org/abs/2403.09636
2. Ecco: Entropy-Aware Cache Compression - https://doi.org/10.1145/3695053.3731024

---

**分析完成时间**：2026-04-12
**分析者**：Metis (Hermes Agent)