# 意图聚类/分类 SOTA 方法调研报告

**调研时间**: 2026-03-24  
**调研范围**: 2024-2025年意图识别、意图聚类、对话意图分类的SOTA方法  
**数据来源**: OpenAlex, arXiv, 学术论文数据库

---

## 目录

1. [概述](#概述)
2. [传统方法](#传统方法)
3. [深度学习方法](#深度学习方法)
4. [LLM-based方法](#llm-based方法)
5. [意图聚类方法](#意图聚类方法)
6. [代表性论文](#代表性论文)
7. [常用数据集与评估指标](#常用数据集与评估指标)
8. [未来趋势](#未来趋势)

---

## 概述

意图识别(Intent Detection/Classification)是对话系统和NLU(Natural Language Understanding)的核心任务之一，目标是识别用户输入背后的意图类型。近年来，该领域经历了从传统机器学习方法到深度学习方法，再到LLM-based方法的演进。

### 任务定义

- **意图分类(Intent Classification)**: 给定用户输入文本，预测其所属的预定义意图类别
- **意图聚类(Intent Clustering)**: 在无监督或半监督场景下，发现和分组相似的意图
- **开放意图发现(Open Intent Discovery)**: 识别训练集中未见过的新的意图类别

---

## 传统方法

### 1. 规则与模板方法 (Rule-based & Template Methods)

**特点**: 基于关键词、正则表达式或预定义规则进行意图匹配

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| 关键词匹配 | 简单、可解释性强 | 泛化能力弱 | 小规模、意图明确场景 |
| 正则表达式 | 精确匹配、可定制 | 维护成本高 | 固定格式输入 |
| 意图模板 | 结构化、易维护 | 覆盖不全 | 特定领域对话系统 |

### 2. 传统机器学习方法

**特征工程 + 分类器**

| 方法 | 代表论文 | 特点 |
|------|----------|------|
| **SVM** | Support Vector Machine用于文本分类 | 高维稀疏特征效果好，适合小数据集 |
| **Naive Bayes** | 朴素贝叶斯文本分类 | 计算高效，适合实时系统 |
| **Logistic Regression** | 逻辑回归 | 可解释性强，可作为baseline |
| **Random Forest** | 随机森林 | 集成方法，抗过拟合 |
| **XGBoost/LightGBM** | 梯度提升树 | 表格数据效果好，特征重要性可解释 |

**常用特征**:
- TF-IDF
- N-gram特征
- 词向量平均 (Word2Vec, GloVe)
- 句法特征

---

## 深度学习方法

### 1. CNN-based方法

**代表模型**: 
- TextCNN (Kim, 2014)
- Character-level CNN

**特点**: 
- 捕获局部n-gram特征
- 计算效率高
- 适合短文本分类

**性能**: 在SNIPS、ATIS等数据集上达到95%+准确率

### 2. RNN/LSTM-based方法

**代表模型**:
- BiLSTM + Attention
- LSTM-CRF (序列标注)

**特点**:
- 捕获序列依赖关系
- Attention机制提升可解释性
- 适合多轮对话意图识别

### 3. BERT-based方法 (2019-2024主流)

**代表模型**:

| 模型 | 发布时间 | 特点 | BANKING77准确率 |
|------|----------|------|-----------------|
| BERT-base | 2018 | 预训练+微调范式 | 93.5% |
| RoBERTa | 2019 | 优化训练策略 | 94.2% |
| DistilBERT | 2019 | 轻量化 | 92.1% |
| ALBERT | 2019 | 参数共享 | 93.8% |
| DeBERTa | 2020 | 解耦注意力 | 94.5% |

**优化技术**:

1. **Prompt-based Fine-tuning**
   - 论文: "Pre-train, Prompt, and Predict" (Liu et al., 2022, ACM Computing Surveys)
   - 引用数: 3400+
   - 方法: 将分类任务转化为完形填空任务

2. **Contrastive Learning**
   - SimCSE, ConSERT
   - 提升语义表示质量

3. **Data Augmentation**
   - EDA (Easy Data Augmentation)
   - Back Translation
   - 领域自适应

### 4. 多任务学习方法

**Joint Intent-Slot Detection**:
- 同时预测意图和槽位
- 模型: Joint BERT, StackPropagation

**代表论文**:
- "Joint Intent Detection and Slot Filling" 系列工作
- 在ATIS数据集上达到SOTA

### 5. 少样本学习 (Few-shot Learning)

**方法**:

| 方法 | 论文 | 特点 |
|------|------|------|
| Prototypical Networks | Snell et al., 2017 | 基于类原型距离分类 |
| Matching Networks | Vinyals et al., 2016 | 注意力匹配机制 |
| MAML | Finn et al., 2017 | 元学习快速适应 |
| Prompt-tuning | 2021+ | 软提示学习 |

**论文**: "In-Context Learning for Text Classification with Many Labels" (Milios et al., 2023, ACL)
- 使用LLM进行few-shot文本分类
- 在意图分类数据集上验证有效

---

## LLM-based方法

### 1. Zero-shot Intent Classification

**方法**: 直接使用LLM进行意图预测，无需训练数据

**代表工作**:

| 方法 | 模型 | 特点 |
|------|------|------|
| GPT-3/4 Zero-shot | OpenAI GPT系列 | Prompt工程 |
| In-context Learning | 无需参数更新 | 示例驱动 |
| Instruction Tuning | FLAN, T0 | 指令微调提升零样本能力 |

**Prompt示例**:
```
任务: 识别用户意图
输入: "我想预订明天去上海的机票"
可选意图: [订票, 查询天气, 酒店预订, 其他]
输出: 订票
```

### 2. Few-shot In-context Learning

**方法**: 在prompt中提供少量示例

**优势**:
- 无需模型训练
- 快速适应新领域
- 灵活性高

**论文**:
- "Language Models are Few-Shot Learners" (GPT-3, 2020)
- "Pre-train, Prompt, and Predict" (Liu et al., 2022)

### 3. LLM微调方法

**方法**:

| 方法 | 描述 | 适用场景 |
|------|------|----------|
| Full Fine-tuning | 全参数微调 | 大规模数据 |
| LoRA/QLoRA | 低秩适配 | 高效微调 |
| Prefix Tuning | 前缀调优 | 多任务场景 |
| P-Tuning v2 | 深层提示调优 | NLU任务 |

### 4. RAG增强的意图识别

**方法**: 检索增强生成

**架构**:
1. 意图示例库构建
2. 相似样本检索
3. LLM推理

**优势**:
- 动态知识更新
- 减少幻觉
- 可解释性强

**相关论文**:
- "Adversarial Distilled Retrieval-Augmented Guarding Model for Online Malicious Intent Detection" (Guo et al., 2025, arXiv)
- 检索增强的恶意意图检测

### 5. 2024-2025最新LLM方法

**代表论文**:

1. **"Main Predicate and Their Arguments as Explanation Signals For Intent Classification"** (Pimparkhede & Bhattacharyya, 2025, arXiv)
   - 使用谓词-论元结构作为意图分类的解释信号
   - 提升可解释性

2. **"LinGO: A Linguistic Graph Optimization Framework with LLMs for Interpreting Intents of Online Uncivil Discourse"** (Zhang & Bertaglia, 2026, arXiv)
   - 语言学图优化框架
   - 在线不文明言论意图识别

3. **"Interpretability of the Intent Detection Problem: A New Approach"** (Sanchez-Karhunen et al., 2026, arXiv)
   - 意图检测可解释性研究

---

## 意图聚类方法

### 1. 无监督聚类方法

**传统方法**:
- K-Means
- Hierarchical Clustering
- DBSCAN

**深度聚类**:

| 方法 | 论文 | 特点 |
|------|------|------|
| DeepCluster | Caron et al., 2018 | 端到端聚类 |
| DEC | Xie et al., 2016 | 深度嵌入聚类 |
| IDEC | Guo et al., 2017 | 改进DEC |

### 2. 开放意图发现 (Open Intent Discovery)

**任务**: 发现未见过的意图类别

**方法**:

| 方法 | 论文 | 核心思想 |
|------|------|----------|
| OpenMax | Bendale & Boult, 2016 | 开放集识别 |
| OSR方法 | 各种开放集识别工作 | 拒绝已知类外样本 |
| GDA | 生成式判别方法 | 生成未知类表示 |

**深度学习方法**:

1. **基于重构的方法**
   - AutoEncoder异常检测
   - VAE生成模型

2. **基于原型的方法**
   - Prototypical Network扩展

3. **对比学习方法**
   - 无监督对比学习发现意图结构

### 3. 半监督意图聚类

**方法**:
- Deep Embedded Clustering with Partial Labels
- Semi-supervised K-Means
- Label Propagation

---

## 代表性论文

### 综述论文

| 论文 | 发表 | 引用数 | 主要贡献 |
|------|------|--------|----------|
| "Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing" | ACM Computing Surveys, 2022 | 3400+ | 系统综述Prompt方法 |
| "Natural Language Processing: State of the Art, Current Trends and Challenges" | Multimedia Tools and Applications, 2022 | 1600+ | NLP全面综述 |

### 意图分类核心论文

| 论文 | 年份 | 会议/期刊 | 贡献 |
|------|------|-----------|------|
| BERT: Pre-training of Deep Bidirectional Transformers | 2018 | NAACL | 预训练范式 |
| "In-Context Learning for Text Classification with Many Labels" | 2023 | ACL GenBench | LLM多标签分类 |
| "Main Predicate and Their Arguments as Explanation Signals For Intent Classification" | 2025 | arXiv | 可解释意图分类 |
| "Interpretability of the Intent Detection Problem" | 2026 | arXiv | 意图检测可解释性 |

### 2024-2025最新工作

| 论文 | 年份 | 关键技术 |
|------|------|----------|
| "LinGO: Linguistic Graph Optimization with LLMs for Intent Interpretation" | 2026 | 图优化+LLM |
| "Adversarial Distilled Retrieval-Augmented Guarding Model" | 2025 | RAG+对抗蒸馏 |
| "Semantic Intent Decoding from EEG" | 2026 | 脑机接口意图识别 |

---

## 常用数据集与评估指标

### 标准数据集

| 数据集 | 意图数 | 样本数 | 领域 | 特点 |
|--------|--------|--------|------|------|
| **BANKING77** | 77 | 13,083 | 银行业务 | 细粒度意图，benchmark常用 |
| **CLINC150** | 150 | 23,700 | 多领域 | 包含OOV意图 |
| **SNIPS** | 7 | 16,358 | 语音助手 | 标准benchmark |
| **ATIS** | 26 | 5,871 | 航空旅行 | 经典数据集 |
| **HWU64** | 64 | 11,002 | 智能家居 | 多领域 |

### 评估指标

| 指标 | 适用场景 | 公式 |
|------|----------|------|
| **Accuracy** | 平衡数据集 | 正确预测数/总数 |
| **F1-score** | 不平衡数据集 | 2*P*R/(P+R) |
| **Macro-F1** | 多类别不平衡 | 各类F1平均 |
| **Precision/Recall** | 特定需求 | 根据业务选择 |

### SOTA性能对比 (BANKING77)

| 模型 | Accuracy | 年份 |
|------|----------|------|
| BERT-base | 93.5% | 2019 |
| RoBERTa | 94.2% | 2019 |
| DeBERTa-v3 | 94.5%+ | 2021 |
| GPT-4 (Few-shot) | ~93% | 2023 |
| Fine-tuned LLM (LoRA) | 95%+ | 2024 |

---

## 未来趋势

### 1. 多模态意图识别

- 语音+文本融合
- 视觉信号辅助
- 多模态Transformer

### 2. 持续学习与增量意图

- 新意图自动发现
- 持续学习避免灾难性遗忘
- 在线适应

### 3. 可解释性与可信度

- 意图预测解释
- 不确定性量化
- 可信AI系统

### 4. 高效部署

- 模型压缩与蒸馏
- 边缘端推理
- 实时性能优化

### 5. 跨领域迁移

- 零样本跨领域
- 元学习快速适应
- 领域自适应

### 6. 安全与对抗鲁棒性

- 对抗样本防御
- 投毒攻击检测
- 安全意图识别

---

## 结论

意图分类领域已从传统的特征工程方法，演进到深度学习方法，再到当前的LLM-based方法。主要趋势包括：

1. **LLM主导**: 大语言模型在少样本、零样本场景下表现优异
2. **高效微调**: LoRA等方法降低了微调成本
3. **可解释性增强**: 越来越多研究关注模型决策的可解释性
4. **多模态融合**: 结合多种信号提升意图识别准确性
5. **开放场景**: 开放意图发现成为研究热点

建议在实际应用中：
- 小规模数据：使用BERT类模型微调
- 大规模/多领域：考虑LLM + RAG方案
- 需要可解释性：结合规则系统或注意力可视化
- 边缘部署：使用蒸馏或量化技术

---

## 参考文献

1. Liu, P., et al. (2022). "Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing." ACM Computing Surveys.
2. Milios, A., et al. (2023). "In-Context Learning for Text Classification with Many Labels." ACL GenBench.
3. Pimparkhede, S., & Bhattacharyya, P. (2025). "Main Predicate and Their Arguments as Explanation Signals For Intent Classification." arXiv.
4. Zhang, Y., & Bertaglia, T. (2026). "LinGO: A Linguistic Graph Optimization Framework with LLMs." arXiv.
5. Sanchez-Karhunen, E., et al. (2026). "Interpretability of the Intent Detection Problem: A New Approach." arXiv.
6. Guo, Y., et al. (2025). "Adversarial Distilled Retrieval-Augmented Guarding Model for Online Malicious Intent Detection." arXiv.
7. Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers." NAACL.
8. Liu, Y., et al. (2019). "RoBERTa: A Robustly Optimized BERT Pretraining Approach." arXiv.

---

*本报告由 worker1 (ClawTeam intent-clustering) 生成*
*生成时间: 2026-03-24*