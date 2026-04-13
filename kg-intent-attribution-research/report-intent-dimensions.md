# 意图属性维度规划研究报告

## 1. 文献调研总结

### 1.1 意图识别分类体系研究

#### 传统意图分类方法
意图识别（Intent Detection）是自然语言理解（NLU）的核心任务，在问答系统和对话系统中起着关键作用。根据系统性文献综述（Zailan et al., 2023），意图检测与槽填充技术主要分为：

| 方法类别 | 代表技术 | 特点 |
|---------|---------|------|
| Pipeline方法 | 基于规则的分类器、传统机器学习 | 分步处理，意图检测与实体提取分离 |
| Joint方法 | BERT-based joint models | 联合建模，意图与槽位相互增强 |
| LLM-based | GPT、LLaMA、ChatGPT | 零样本/少样本能力强，泛化性好 |

#### 混合架构趋势（Voiceflow, 2024）
最新的研究表明，混合NLU/LLM架构在大型意图分类数据集上表现优异：
- 使用编码器模型找到top-10候选意图
- 使用LLM进行最终分类判断
- 在HWU、Banking77、CLINC150等数据集上验证有效

### 1.2 问答系统问题类型分类

#### 问题类型层次化分类（Jurafsky, 2019）
Stanford QA框架将问题分为以下层次结构：

| 顶层类型 | 子类型 | 回答特征 |
|---------|--------|---------|
| **Factoid** | Person/Location/Date/Number/Organization | 简短事实性答案 |
| **Confirmation** | Yes/No | 二值确认 |
| **List** | 多实体列举 | 实体列表 |
| **Causal** | Why/How | 解释性答案 |
| **Hypothetical** | What if | 条件推理答案 |
| **Complex** | Compositional/Multi-hop | 组合推理答案 |

#### Bloom认知层次的应用
Bloom Taxonomy（认知层次）在问题分类中的应用：
1. **Knowledge** - 简单记忆检索
2. **Comprehension** - 理解解释
3. **Application** - 应用执行
4. **Analysis** - 分析分解
5. **Synthesis** - 综合构建
6. **Evaluation** - 评价判断

### 1.3 多意图对话系统研究

#### 多意图层次化处理（ACM, 2021）
"A Unified Dialogue Management Strategy for Multi-intent Dialogue" 提出层次化结构处理多意图：
- **Intent Graph** - 意图节点及依赖关系图
- **Hierarchical DM** - 分层对话管理策略
- **Multi-label Classification** - 支持同时多个意图标签

#### 意图组合模式（arXiv, 2024）
BlendX/MixX研究识别了多意图组合类型：
- **Explicit Conjunction** - "X and Y" 显式连接
- **Implicit Conjunction** - 隐含多个目标
- **Coreference** - 代词跨意图引用
- **Omission** - 略述后续意图

---

## 2. 现有分类体系对比表

### 2.1 NLU意图识别数据集对比

| 数据集 | 意图数量 | 领域 | 特点 | 层次结构 |
|--------|---------|------|------|---------|
| **SNIPS** | 7 | 通用助手 | 语音命令场景 | 扁平 |
| **Banking77** | 77 | 银行业务 | 高度细粒度 | 扁平+领域分组 |
| **CLINC150** | 150 | 多领域 | 10领域×15意图 | 扁平+领域分层 |
| **HWU64** | 64 | 家庭助手 | 21领域×多意图 | 层次化 |
| **ATIS** | 26 | 航旅 | 槽位丰富 | 扁平 |
| **MTOP** | 113 | 多语言 | 跨语言意图 | 层次化 |

### 2.2 KGQA基准数据集对比

| 数据集 | 问题类型 | KG规模 | 跳数范围 | 意图复杂度 |
|--------|---------|--------|---------|-----------|
| **WebQuestionsSP** | 事实型+复杂型 | Freebase | 1-2 hop | 单意图为主 |
| **ComplexWebQuestions** | 组合型+条件型 | Freebase | 1-4 hop | 高复杂度 |
| **MetaQA** | 电影领域 | WikiMovies | 1-3 hop | 明确跳数标签 |
| **QALD-9** | 多类型 | DBpedia | 1-3 hop | SPARQL意图多样 |
| **LC-QuAD** | 复杂型 | DBpedia | 2-3 hop | 多关系组合 |

### 2.3 问题类型分类框架对比

| 框架 | 层级数 | 分类维度 | 适用场景 | KG适配性 |
|------|-------|---------|---------|---------|
| **GQCC Grammar-based** | 2层 | 语法特征 | 通用QA | 低 |
| **Li & Roth Taxonomy** | 2层 | 答案类型 | TREC QA | 中 |
| **CQA Classification** | 3类 | 语义复杂度 | KGQA | 高 |
| **Bloom Cognitive** | 6层 | 认知难度 | 教育/评估 | 中 |
| **IBM Watson LAT** | 层次化 | 答案类型+置信度 | Jeopardy | 中 |

---

## 3. KG意图属性设计方案

### 3.1 设计原则

基于文献调研和KG特殊性分析，KG RAG意图属性设计应遵循以下原则：

1. **层次化可扩展** - 支持从粗粒度到细粒度的意图分解
2. **KG结构映射** - 意图属性与三元组结构(E-R-E)对应
3. **跳数显式化** - 多跳推理需要明确的跳数维度
4. **组合性支持** - 支持多意图组合的表达
5. **答案类型绑定** - 意图与预期答案类型关联

### 3.2 KG意图类型层次结构

```
KGIntent
├── QueryIntent (查询意图)
│   ├── EntityQuery (实体查询)
│   │   ├── SingleEntity (单实体检索)
│   │   ├── EntityList (实体列举)
│   │   └── EntityRanking (实体排序)
│   ├── RelationQuery (关系查询)
│   │   ├── DirectRelation (直接关系)
│   │   ├── IndirectRelation (间接关系)
│   │   └── RelationPath (关系路径)
│   ├── AttributeQuery (属性查询)
│   │   ├── SingleAttr (单一属性)
│   │   ├── MultiAttr (多属性组合)
│   │   └── AttrAggregation (属性聚合)
│   └── ComplexQuery (复杂查询)
│       ├── Compositional (组合型)
│       ├── Comparative (比较型)
│       ├── Conditional (条件型)
│       └── Temporal (时序型)
├── ActionIntent (操作意图)
│   ├── KGNavigation (图谱导航)
│   ├── InfoExpansion (信息扩展)
│   └── ReasoningTrace (推理追踪)
└── MetaIntent (元意图)
    ├── Clarification (澄清请求)
    ├── ContextRef (上下文引用)
    ├── OutOfScope (超出范围)
    └── Ambiguous (意图模糊)
```

### 3.3 KG意图属性维度定义

#### 核心属性维度

| 维度 | 定义 | 取值范围 | 说明 |
|------|------|---------|------|
| **intent_type** | 意图类型 | QueryIntent/ActionIntent/MetaIntent | 顶层意图分类 |
| **query_subtype** | 查询子类型 | EntityQuery/RelationQuery/AttributeQuery/ComplexQuery | 二级分类 |
| **hop_count** | 跳数 | 1/2/3/.../N | KG遍历深度 |
| **answer_type** | 答案类型 | Entity/Relation/Attribute/List/Boolean/Number/Explanation | 预期输出类型 |
| **complexity_level** | 复杂度 | Simple/Medium/Complex | 推理复杂度等级 |
| **is_compositional** | 组合性 | true/false | 是否为多意图组合 |
| **constraint_type** | 约束类型 | None/Temporal/Spatial/Quantitative/Logical | 约束条件类型 |
| **entity_count** | 实体数 | 0/1/2/.../N | 问题中提及实体数量 |

#### 补充属性维度

| 维度 | 定义 | 取值示例 | 说明 |
|------|------|---------|------|
| **focus_entity_type** | 焦点实体类型 | Person/Location/Organization... | 关注的实体类别 |
| **reasoning_type** | 推理类型 | Direct/Transitive/Comparative/Temporal | 推理模式 |
| **lexical_pattern** | 词法模式 | "Who is..." / "How many..." | 典型问法模板 |
| **confidence** | 意图置信度 | 0.0-1.0 | 意图识别置信分数 |

### 3.4 KG适配分析

#### KG三元组结构与意图映射

| KG元素 | 对应意图属性 | 示例 |
|--------|-------------|------|
| **Entity (主体)** | focus_entity, entity_count | "Who is the CEO of Apple?" → focus_entity=Apple |
| **Relation** | relation_type, hop_count | "Who founded Microsoft?" → relation=founder, hop=1 |
| **Entity (客体/答案)** | answer_type | "Steve Jobs" → answer_type=Entity |
| **Attribute** | query_subtype=AttributeQuery | "What is the population of Tokyo?" |
| **Multi-hop Path** | hop_count, reasoning_type | "Who is the spouse of the director of Inception?" → hop=2 |

#### 多跳查询的意图组合性

多跳KGQA的意图具有组合特征：

| 跳数 | 意图特征 | 示例问题 |
|------|---------|---------|
| **1-hop** | 单意图，直接关系 | "Who directed Titanic?" |
| **2-hop** | 组合意图，链式推理 | "Who is the spouse of the director of Titanic?" |
| **3-hop** | 多意图组合，路径推理 | "What movies starred the spouse of the director of Inception?" |
| **N-hop** | 深层组合，复杂路径 | "Find awards won by movies directed by children of Nobel laureates" |

---

## 4. 形式化定义

### 4.1 KGIntent JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "KGIntent",
  "description": "Knowledge Graph Question Answering Intent Schema",
  "type": "object",
  "required": ["intent_type", "query_subtype", "hop_count", "answer_type"],
  "properties": {
    "intent_id": {
      "type": "string",
      "description": "唯一意图标识符"
    },
    "intent_type": {
      "type": "string",
      "enum": ["QueryIntent", "ActionIntent", "MetaIntent"],
      "description": "顶层意图类型"
    },
    "query_subtype": {
      "type": "string",
      "enum": ["EntityQuery", "RelationQuery", "AttributeQuery", "ComplexQuery"],
      "description": "查询子类型"
    },
    "hop_count": {
      "type": "integer",
      "minimum": 1,
      "maximum": 10,
      "description": "KG遍历跳数"
    },
    "answer_type": {
      "type": "string",
      "enum": ["Entity", "Relation", "Attribute", "List", "Boolean", "Number", "Explanation", "GraphPath"],
      "description": "预期答案类型"
    },
    "complexity_level": {
      "type": "string",
      "enum": ["Simple", "Medium", "Complex"],
      "default": "Simple",
      "description": "推理复杂度"
    },
    "is_compositional": {
      "type": "boolean",
      "default": false,
      "description": "是否为组合意图"
    },
    "sub_intents": {
      "type": "array",
      "items": { "$ref": "#" },
      "description": "子意图列表（组合意图时使用）"
    },
    "constraint_type": {
      "type": "string",
      "enum": ["None", "Temporal", "Spatial", "Quantitative", "Logical", "Multiple"],
      "default": "None",
      "description": "约束条件类型"
    },
    "entity_count": {
      "type": "integer",
      "minimum": 0,
      "description": "问题中实体数量"
    },
    "focus_entity_type": {
      "type": "string",
      "description": "焦点实体类型"
    },
    "reasoning_type": {
      "type": "string",
      "enum": ["Direct", "Transitive", "Comparative", "Temporal", "Boolean", "Aggregation"],
      "description": "推理模式类型"
    },
    "lexical_pattern": {
      "type": "string",
      "description": "典型问法模板"
    },
    "confidence": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "default": 0.8,
      "description": "意图识别置信度"
    },
    "kg_elements": {
      "type": "object",
      "properties": {
        "entities": {
          "type": "array",
          "items": { "type": "string" },
          "description": "涉及的KG实体列表"
        },
        "relations": {
          "type": "array",
          "items": { "type": "string" },
          "description": "涉及的KG关系列表"
        },
        "attributes": {
          "type": "array",
          "items": { "type": "string" },
          "description": "涉及的属性列表"
        }
      },
      "description": "KG元素提取结果"
    },
    "query_structure": {
      "type": "object",
      "properties": {
        "pattern": {
          "type": "string",
          "enum": ["E-R-E", "E-R-E-R-E", "Multi-Path", "Aggregation", "Filter"],
          "description": "查询结构模式"
        },
        "depth": {
          "type": "integer",
          "description": "查询树深度"
        }
      },
      "description": "查询结构信息"
    }
  },
  "additionalProperties": true
}
```

### 4.2 意图标注示例

#### 示例1：单跳实体查询
```json
{
  "intent_id": "kg_intent_001",
  "intent_type": "QueryIntent",
  "query_subtype": "EntityQuery",
  "hop_count": 1,
  "answer_type": "Entity",
  "complexity_level": "Simple",
  "is_compositional": false,
  "constraint_type": "None",
  "entity_count": 1,
  "focus_entity_type": "Company",
  "reasoning_type": "Direct",
  "lexical_pattern": "Who is the CEO of X?",
  "confidence": 0.95,
  "kg_elements": {
    "entities": ["Apple Inc."],
    "relations": ["CEO"],
    "attributes": []
  },
  "query_structure": {
    "pattern": "E-R-E",
    "depth": 1
  }
}
```

#### 示例2：两跳组合查询
```json
{
  "intent_id": "kg_intent_002",
  "intent_type": "QueryIntent",
  "query_subtype": "ComplexQuery",
  "hop_count": 2,
  "answer_type": "Entity",
  "complexity_level": "Medium",
  "is_compositional": true,
  "sub_intents": [
    {
      "intent_type": "QueryIntent",
      "query_subtype": "RelationQuery",
      "hop_count": 1,
      "lexical_pattern": "Who directed X?"
    },
    {
      "intent_type": "QueryIntent",
      "query_subtype": "RelationQuery",
      "hop_count": 1,
      "lexical_pattern": "Who is the spouse of X?"
    }
  ],
  "constraint_type": "None",
  "entity_count": 1,
  "focus_entity_type": "Person",
  "reasoning_type": "Transitive",
  "lexical_pattern": "Who is the spouse of the director of X?",
  "confidence": 0.88,
  "kg_elements": {
    "entities": ["Inception"],
    "relations": ["director", "spouse"],
    "attributes": []
  },
  "query_structure": {
    "pattern": "E-R-E-R-E",
    "depth": 2
  }
}
```

#### 示例3：聚合属性查询
```json
{
  "intent_id": "kg_intent_003",
  "intent_type": "QueryIntent",
  "query_subtype": "AttributeQuery",
  "hop_count": 1,
  "answer_type": "Number",
  "complexity_level": "Simple",
  "is_compositional": false,
  "constraint_type": "Quantitative",
  "entity_count": 1,
  "focus_entity_type": "City",
  "reasoning_type": "Aggregation",
  "lexical_pattern": "How many people live in X?",
  "confidence": 0.92,
  "kg_elements": {
    "entities": ["Tokyo"],
    "relations": [],
    "attributes": ["population"]
  },
  "query_structure": {
    "pattern": "E-Attr-Agg",
    "depth": 1
  }
}
```

### 4.3 意图粒度设计建议

| 粒度层级 | 适用场景 | 意图属性完备度 |
|---------|---------|---------------|
| **文档级** | RAG索引构建、文档分类 | intent_type, query_subtype |
| **段落级** | 检索增强、相关性匹配 | + hop_count, answer_type |
| **Chunk级** | 精细意图标注、训练数据 | 完整Schema属性 |
| **Token级** | 槽位提取、实体识别 | kg_elements, lexical_pattern |

---

## 5. 结论与展望

### 5.1 主要贡献

1. **提出KG意图层次结构** - 适配KG三元组结构的3层意图分类体系
2. **定义核心属性维度** - 8个核心维度+4个补充维度的完整属性集
3. **形式化JSON Schema** - 可直接用于KGQA系统实现
4. **组合意图支持** - 支持多跳查询的意图分解表达

### 5.2 后续研究方向

1. **意图权重体系** - 设计各维度权重用于检索排序
2. **意图标注工具** - 开发KG数据自动意图标注工具
3. **跨域意图泛化** - 研究意图分类的领域迁移能力
4. **动态意图调整** - 支持对话上下文中的意图演化

---

## 参考文献

1. Zailan, A. S. M., et al. (2023). "State of the Art in Intent Detection and Slot Filling for Question Answering System: A Systematic Literature Review." IJACSA 14(11).
2. Voiceflow (2024). "Benchmarking hybrid LLM classification systems."
3. Jurafsky, D. (2019). "Question Answering." SLP3 Chapter 25, Stanford University.
4. ACM (2021). "A Unified Dialogue Management Strategy for Multi-intent Dialogue."
5. Larson, S., et al. (2019). "A Survey of Intent Classification and Slot-Filling Datasets for Task-Oriented Dialogue."
6. Talmor, A., et al. (2018). "The Web as a Knowledge-Base for Answering Complex Questions."
7. Zhang, Y., et al. (2025). "Diagnosing and Addressing Pitfalls in KG-RAG Datasets." arXiv:2505.23495.
8. Wang, H., et al. (2024). "Multi-hop Question Answering over Knowledge Graphs using Large Language Models." arXiv:2404.19234.
9. Neo4j (2024). "How to Improve Multi-Hop Reasoning With Knowledge Graphs and LLMs."
10. Bast, H., et al. (2015). "More Question Answering on Knowledge Bases."

---

*报告完成时间: 2026-04-01*
*研究时长: 约25分钟*