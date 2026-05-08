# Text2SQL SOTA 技术调研报告

> 调研时间: 2026-05-07 | 模型: deepseek-v4-flash | 方法: HermesTeam 并行 Delegate

---

## 一、概览

Text2SQL 的核心目标是将自然语言问题转换为可执行的 SQL 查询。2023-2026 年间，该领域经历了快速演进：

- **2023**: DIN-SQL 提出分解范式，C3 探索 zero-shot
- **2024**: DAIL-SQL 优化 few-shot 选择，MAC-SQL 引入多 Agent 协作，CHESS 强化检索+剪枝
- **2025**: XiYan-SQL 多生成器集成达 Spider 89.65%，DTS-SQL 证明小模型+分解可接近大模型性能
- **2026**: 主流趋势是多 Agent 协作、多生成器集成、训练数据质量校正

当前标准 Pipeline 由四个阶段构成：

```
[NL Question + DB Schema]
        │
        ▼
┌─────────────────────────┐
│ 1. Schema Linking       │  ← 表筛选、列匹配、FK推理
│    (数据理解)            │
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│ 2. SQL Generation       │  ← 分解/Skeleton/CoT/多候选
│    (SQL转换)             │
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│ 3. Verification         │  ← 执行验证/Self-Correction
│    (正确性保证)          │
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│ 4. Selection/Ranking    │  ← Voting/Re-ranking
│    (最优选择)            │
└─────────────────────────┘
```

---

## 二、Schema Linking — 数据理解

### 2.1 Pipeline 层级

典型的 Schema Linking 分三级递进：

| 层级 | 目标 | 方法 | 代表工作 |
|------|------|------|---------|
| DB Selection | 选目标数据库 | Embedding 召回 | LinkAlign (EMNLP 2025) |
| Table Selection | 选相关表 | Cross-encoder/LLM/RAG | RESDSQL (AAAI 2023), DFIN-SQL |
| Column Selection | 选相关列 | 多策略混合 | CHESS, DTS-SQL |

### 2.2 筛选策略对比

| 方法 | 精度 | 速度 | 可扩展性 | 代表 |
|------|:----:|:----:|:--------:|------|
| Embedding 粗筛 | 中 | 高 | 高 | CHESS — LSH + 向量库 |
| Cross-Encoder 精排 | 高 | 中 | 低（O(N)） | RESDSQL |
| LLM 自判断 | 高 | 低 | 低（Token 成本+超窗） | DIN-SQL, MAC-SQL |
| RAG+LLM 混合 | 高 | 中 | 高 | DFIN-SQL |
| Agentic 迭代探索 | 极高 | 低 | 极高 | AutoLink (AAAI 2025) — 97.4% Recall |
| Graph 编码 | 高 | 低 | 中 | RAT-SQL, SchemaGraphSQL |

### 2.3 值检索 (Value Retrieval)

当问题包含具体值（人名、产品名、日期），需要从数据库中检索对应值：

| 方法 | 原理 | 适用场景 |
|------|------|---------|
| Fuzzy Match | Levenshtein/pg_trgm 模糊搜索 | 小表 |
| Embedding (Vector DB) | LSH + 向量库检索 cell value | 大表（百万级） |
| DB SQL 扫描 | 执行 LIKE/WHERE 查询发现值 | 实时准确但慢 |
| Value Hints | 采样列值作为上下文提示 | 帮助 LLM 理解列语义 |

### 2.4 FK 关系推理

| 方法 | 原理 | 代表 |
|------|------|------|
| 显式 Schema 图编码 | Relational Graph Attention | RAT-SQL (ACL 2020) |
| 路径搜索 | Dijkstra/BFS 找最短 join 路径 | SchemaGraphSQL (2025) |
| LLM 隐式推理 | FK 描述放 prompt | DIN-SQL, MAC-SQL |
| Steiner Tree 优化 | 图优化找最小连接子图 | Scaling Text2SQL (2025) |

### 2.5 关键论文速览

| 论文 | 会议/年份 | Schema Linking 方法 |
|------|:---------:|---------------------|
| DIN-SQL | NeurIPS 2023 | LLM 自判断 schema linking + query classification |
| RESDSQL | AAAI 2023 | Cross-encoder 排序 + 骨架解析解耦 |
| DAIL-SQL | VLDB 2024 | 继承 RAT-SQL 的 linking + 动态 few-shot |
| CHESS | ICML WS 2025 | 三级检索（关键词+LSH+向量库）+ 自适应剪枝 |
| MAC-SQL | COLING 2025 | Selector Agent 将大 DB 压缩为子数据库 |
| DTS-SQL | EMNLP 2024 | 两阶段微调（linking 模型 + 生成模型） |
| X-SQL | 2025 | X-Linking: 专门 SFT 微调的开源 SOTA |
| CodeS | SIGMOD 2024 | SQL-centric 预训练 + 多粒度 Schema 编码 |
| AutoLink | AAAI 2025 | Agent 迭代探索（DB Environment + Vector Store） |
| LinkAlign | EMNLP 2025 | Embedding 粗筛 + 双向对齐，多 DB 场景 |

### 2.6 关键争论: "The Death of Schema Linking?"

**TLR/NeurIPS 2024**: 新一代 reasoning LLM 在 schema 完整输入时可自行识别相关列，显式 schema linking 不再必要。结论：

- **schema 能放入 context window → 不筛比筛好**（Recall 更高）
- **schema 超窗 → 仍需剪枝**

> ⚠️ 实践建议: 现代 LLM（GPT-4, Claude, DeepSeek）的 schema 理解能力足够强，优先考虑全 schema 输入，仅在超窗或性能问题时才启用显式 schema linking。

---

## 三、SQL 生成 Pipeline

### 3.1 架构对比

| 架构 | 原理 | 代表 | 优点 | 缺点 |
|------|------|------|------|------|
| End-to-End | 一次生成完整 SQL | C3, 基础 GPT-4 | 简单 | 复杂查询差 |
| Decomposed | 分阶段生成 | DIN-SQL, DTS-SQL | 每阶段更聚焦 | 误差累积 |
| Multi-Agent | 多 Agent 协作 | MAC-SQL, CHESS | 专业化分工 | 流程复杂 |
| Multi-Generator | 多模型集成 | XiYan-SQL | 精度最高 | 成本高 |

### 3.2 中间表示 (Intermediate Representation)

| IR 方法 | 原理 | 优点 | 缺点 |
|---------|------|------|------|
| Skeleton-first | 先骨架后填充 | 保证结构正确 | 骨架分类错误→级联失败 |
| Step-by-step CoT | 分析→条件提取→构建 | 可解释性好 | Token 消耗大 |
| Sub-query Decomposition | 子问题拆解 | 降低单步难度 | 依赖管理复杂 |

### 3.3 候选生成与排序

| 方法 | 原理 | 代表 |
|------|------|------|
| Beam Search | 多种部分 SQL 并行扩展 | Execution-Guided Decoding (2018) |
| Temperature Sampling | 多候选随机采样 | DAIL-SQL, C3 |
| Execution-based | 选执行结果最合理的 | C3 Self-Consistency |
| Confidence-based | 选 LLM 置信度最高 | Self-consistency |
| Majority Voting | 投票选最频繁结果 | DAIL-SQL, C3 |
| Model Re-ranker | 训练专门排序模型 | XiYan-SQL |

### 3.4 Prompting 策略

**DAIL-SQL 的 Masked Similarity**（核心发现）:

1. Question Representation: 仅用 question > question+SQL
2. Example Selection: Masked similarity > full similarity > random
3. Example Organization: Question-SQL pair > 仅 SQL > 全信息

| 策略 | 原理 | 代表 |
|------|------|------|
| BM25/TF-IDF | 关键词匹配 | 经典方法 |
| Sentence-BERT | 语义相似度 | 通用 |
| Masked Similarity | 去 Schema token 后算相似度 | DAIL-SQL |
| RoBERTa tokenizer | 特殊 token 化 | CHESS |

---

## 四、正确性与稳定性保证机制

### 4.1 执行反馈 (Execution-guided Decoding)

```
生成部分SQL → 在DB上部分执行 → 根据结果剪枝/保留
```

- **代表**: Execution-Guided Decoding (Wang et al., 2018)
- **原理**: 解码过程中对部分生成的 SQL 执行，过滤掉执行错误的候选
- **效果**: WikiSQL EX 83.8%（当时 SOTA）

### 4.2 自修正 (Self-Correction / Self-Debugging)

```
SQL生成 → 执行 → 错误捕获 → 分析 → 修正 → 重执行
```

- **DIN-SQL Self-Correction**: 执行→错误注入 prompt→LLM 修正→+2-3%
- **MAC-SQL Refiner Agent**: 最多 N 轮修正
- **MAGIC (AAAI 2025)**: 自动生成修正指南，替代人工规则

**关键挑战**:

| 挑战 | 说明 | 缓解 |
|------|------|------|
| 确认偏差 | LLM 倾向保留原答案 | 提供外部错误信息 |
| 修正后精度下降 | 可能改对为错 | 限制修正轮次 max 2-3 |
| Token 开销大 | 每轮需重新推理 | 仅失败时修正 |

### 4.3 正确性保障方法对比

| 方法 | 保障类型 | 成本 | 提升效果 |
|------|---------|:----:|:--------:|
| Self-Consistency Voting | 统计鲁棒性 | 低 | +2-5% |
| Execution Verification | 语法+语义 | 中 | +3-8% |
| Self-Correction | 错误修复 | 高 | 依赖错误信息质量 |
| Test Suite Validation | 数据分布鲁棒性 | 高 | 最可靠 |
> ⚠️ 实践启示: 与其追求复杂的模型架构，优先确保训练数据的准确性可能是性价比最高的提升手段。

---

## 五、Benchmark 评估体系

### 5.1 四大 Benchmark 对比

| 维度 | WikiSQL | Spider 1.0 | BIRD | Spider 2.0 |
|------|---------|-----------|------|------------|
| **发布年** | 2017 | 2018 | 2023 (NeurIPS) | 2024 (ICLR 2025) |
| **问题-SQL 对** | 80,654 | 10,181 | 12,751 | 632 (workflow级) |
| **DB 数** | 26,521 (单表) | 200 (多表, 跨域) | 95 (大型, 33.4GB) | 数十企业级 DB |
| **SQL 复杂度** | 简单(SELECT-WHERE) | 中-复杂(JOIN,子查询) | 复杂(CTE,窗口函数) | 极高(100+行,多步) |
| **表数/DB** | 1 | 5-10 | 10-50 | 100-1000+列 |
| **指标** | EA | Test Suite, ESM | EX, R-VES | Success Rate |
| **脏数据** | 无 | 无 | 有 | 有 |
| **外部知识** | 无 | 无 | 有(文档/业务规则) | 有 |

### 5.2 评价指标

| 指标 | 含义 | 优点 | 缺点 |
|------|------|------|------|
| **EX/EA** | 执行结果对比 | 宽容(等价写法通过) | 巧合相同可能漏检 |
| **ESM** | AST 结构完全匹配 | 严格检查逻辑 | 过于严格 |
| **Test Suite** | 多 DB 多 case 验证 | 鲁棒 | 构造成本高 |
| **R-VES** | BIRD 独有: 正确+效率 | 兼顾效率 | 阈值需定制 |
| **SR** | Spider 2.0: 全流程成功 | 端到端 | 粒度粗 |

### 5.3 BIRD vs Spider 核心差异

1. **脏数据**: 列名 `trn_val_01` vs `name`，含 NULL/格式不一致
2. **外部知识**: ~30% 问题需业务文档（如 OWNER="拥有者账户"）
3. **效率维度**: R-VES 奖励高效 SQL
4. **标注错误**: BIRD 开发集 52.8% 标注错误（CIDR 2026 论文发现）

### 5.4 Spider 2.0 新挑战

- **多步工作流**: 非单条 SQL，需 ETL+调试+多步分析
- **极大规模 Schema**: 1000+ 列，全量放不进 prompt
- **多方言**: BigQuery/Snowflake/DuckDB
- **长上下文**: 平均 84K tokens（LiveSQLBench）
- **代码仓库**: dbt 项目结构理解

### 5.5 当前 SOTA 准确率

**Spider 1.0 (Test Suite)**:

| 排名 | 模型 | 准确率 |
|:----:|------|:------:|
| 1 | MiniSeek | **91.2%** |
| 2 | DAIL-SQL+GPT-4+SC | 86.6% |
| 3 | DIN-SQL+GPT-4 | 85.3% |

**BIRD (Execution Accuracy)**:

| 模型 | Dev EX | Test EX |
|------|:------:|:-------:|
| 人类基线 | — | 92.96% |
| Arctic-Text2SQL-R1-32B | 71.83% | — |
| JD-UNION-Text2sql-32B | — | 70.93% |
| RSL-SQL+GPT-4o | 67.21% | 68.70% |
| GPT-4 baseline | 46.35% | 54.89% |

**Spider 2.0-Snow (Success Rate)**:

| 模型 | Score |
|------|:-----:|
| Sentinel Agent v2 Pro | 96.70% |
| QUVI-3+Gemini-3-pro | 94.15% |
| DAIL-SQL+GPT-4o baseline | 2.20% |

---

## 六、行业落地实践

### 6.1 主要企业方案

| 产品 | 开源 | Stars | 技术栈 | 核心卖点 |
|------|:----:|:-----:|--------|---------|
| **Vanna.ai** | ✅ MIT | ~20k | Python, RAG, 向量库 | 轻量组件库, pip install 即用 |
| **WrenAI** | ✅ | ~12k | 语义层引擎, dbt | 完整 GenBI 平台, 安全治理 |
| **Dataherald** | ✅ | ~3.6k | LangChain, 微调 | 双 Agent(RAG→FT), NL-to-SQL API |
| **DB-GPT** | ✅ | ~13k | 多 Agent, AWEL | 多 Agent 编排, 复杂分析 |
| **Numbers Station** | 模型开源 | ~1.5k | NSQL 系列模型 | 自研 SQL 专用模型 |
| **Outerbase** | ❌ 商业 | — | EZQL, BI 可视化 | 全栈 NL→图表, 一站式 |
| **AskYourDatabase** | ❌ 商业 | — | 多 LLM, 可视化 | 对话式查询, 非技术用户 |

### 6.2 开源 vs 商业

| 维度 | 开源路线 | 商业路线 |
|------|---------|---------|
| 初始成本 | 免费自建，需工程人力 | 按量付费，即开即用 |
| 定制性 | 完全可控 | 产品边界限制 |
| 安全合规 | 需自建 | 原生内置(RBAC,审计) |
| 功能完整性 | 组件化，需组装 | 开箱即用 |
| 适合场景 | 有自研能力团队 | 快速验证/非技术团队 |

### 6.3 生产环境挑战

**挑战 1: 保证 SQL 正确性 → 分层验证 Pipeline**

```
语法验证 → Schema绑定 → 语义验证 → 执行验证 → 业务验证
```
完整 pipeline 可将错误率从 20-30% 降至 2-5%

**挑战 2: Schema 过大 → 分层检索+增量暴露**
- 向量化表名+列描述做预检索
- 按业务域分组(finance/HR/sales)
- 首轮只暴露 5-8 张表，出错扩大范围

**挑战 3: SQL 安全**

| 风险 | 控制手段 |
|------|---------|
| 危险 SQL(DROP等) | 白名单只允许 SELECT |
| Schema 幻觉 | Catalog 验证 |
| 权限绕过 | 列级权限 + RBAC |
| Prompt 注入 | 验证 SQL 而非 prompt |
| 高消耗查询 | 超时+最大扫描量+LIMIT |
| 多租户泄露 | 租户ID自动注入 WHERE |

**挑战 4: 多轮对话**
- 结构化管理: {上一轮 SQL, 解释, 确认假设, schema 子集}
- LLM 自动压缩长对话
- 增量修正: 保留上下文做差异更新

### 6.4 生产正确性保证最佳实践

| 实践 | 说明 | 优先级 |
|------|------|:----:|
| 只读副本/沙箱 | 杜绝写操作 | P0 |
| 分层验证 Pipeline | 语法→Schema→语义→执行→业务 | P0 |
| 语义层先行 | 定义业务词汇表，消除歧义 | P0 |
| Schema Linking 优化 | 50+ 表先检索后生成 | P1 |
| 成本与资源限制 | 超时30s, 最大10K行, 扫描量上限 | P1 |
| 用户反馈闭环 | 点赞/点踩 → 示例库持续学习 | P2 |
| Benchmark 追踪 | 定期回归测试 | P2 |
| 灰度发布+监控 | 新模型10%流量灰度 | P2 |
| 微调+Prompt混合 | 高频量化模型，低频 RAG | P3 |
| 人工审核兜底 | 高价值查询强制人工 | P3 |

---



## 七、微调 (Fine-tuning) 路线

### 7.1 专用 SQL LLM

| 模型 | 基座 | 参数量 | Spider Dev EX | 方法 |
|------|------|:------:|:------------:|------|
| SQL-PaLM | PaLM 2 | ~540B | ~84% | Execution Feedback 微调 |
| SQL-Coder | StarCoder | 15B | ~82% | SQL 代码微调 |
| CodeS | CodeGen | 1B/3B/7B/15B | ~85% (7B) | SQL-centric 增量预训练 |
| SQL-Llama | Code Llama | 7B | (MAC-SQL 配套) | 指令微调 |

### 7.2 微调 vs Prompting 选择

| 维度 | 微调 (SFT) | Prompting (ICL) |
|------|-----------|-----------------|
| 准确率上限 | 更高 (XiYan 89.65%) | 依赖基础模型 |
| 部署成本 | 需要 GPU 推理 | 仅 API 调用费 |
| 数据需求 | 大量标注 (Spider 10K+) | 少量示例 |
| 泛化性 | 可能过拟合 | Schema 无关性好 |
| Token 效率 | 低（模型参数多） | 中（推理 token 少） |

### 7.3 关键发现: 训练数据质量是根本瓶颈

**SQLDriller (SIGMOD 2024)** 发现 Spider/BIRD 中 **>30% 训练数据标注错误**。清洗后模型准确率提升最多 **13.6%**。

> ⚠️ 实践启示: 与其追求复杂的模型架构，优先确保训练数据的准确性可能是性价比最高的提升手段。

---

## 八、2025-2026 最新趋势

### 趋势 1: 多 Agent 协作成为主流

MAC-SQL → CHESS → MAGIC: 从单 LLM → 专业化 Agent 协作（Selector/Decomposer/Refiner/Validator）

### 趋势 2: 多生成器集成

XiYan-SQL (Spider 89.65%, BIRD 75.63%): 多个 fine-tuned 模型比同一模型多次采样效果更好

### 趋势 3: 训练数据质量校正

SQLDriller 发现 30%+ 错误标注，清洗后提升 13.6%

### 趋势 4: 小模型 + 高质量分解 = 大效果

- DTS-SQL + DeepSeek 7B = 85.5% EX
- SLM-SQL 0.6B 模型超过 15B CodeS

### 趋势 5: RAG 成为 Schema Linking 基础设施

Embedding RAG 替代传统规则匹配，作为粗筛阶段的标准方案

### 趋势 6: Agentic 探索策略

AutoLink 通过执行 SQL 与真实数据库交互，逐步发现 Schema 信息

---

## 九、场景化推荐实践方案

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| **企业生产 (高准确率)** | Multi-Agent (CHESS/MAC-SQL) + Execution Verification + Self-Correction | 多层保障 |
| **学术研究 (最高 SOTA)** | XiYan-SQL (多生成器集成) | Spider 89.65%, BIRD 75.63% |
| **低资源部署** | DTS-SQL + DeepSeek/Codes 7B + LoRA SFT | 小模型 + 分解 = 高性价比 |
| **快速原型** | DAIL-SQL + GPT-4/DeepSeek | 86.6% EX, 实现简单 |
| **正确性优先** | SQLDriller 数据清洗 + Execution Verification Loop + Test Suite | 最严格保障 |
| **实时交互场景** | Vanna.ai 路线 + 对话历史 + 用户确认 | 多轮对话友好 |

### 核心技术要点总结表

| 技术维度 | 核心结论 |
|----------|---------|
| 架构选择 | 分解式全面优于 End-to-End |
| Schema Linking | 小 schema 全输入，大 schema 多路径检索 |
| 正确性信号 | 执行结果 > LLM 自省 > Voting |
| Pipeline 标准 | Schema Linking → SQL 生成 → 验证 → 选择 |
| 成本优化 | 小模型 + 高质量分解可接近大模型 |
| 数据质量 | 训练数据质量是最根本的瓶颈 |
| 主流方法 | Multi-Agent 协作是 2025-2026 主流 |

---

### 代表性论文速查表

| 论文 | 会议 | 年份 | 核心贡献 |
|------|------|:----:|---------|
| DIN-SQL | NeurIPS | 2023 | 四步分解 + Self-Correction |
| C3 | arXiv | 2023 | Zero-shot ChatGPT + Skeleton |
| DAIL-SQL | VLDB | 2023 | Masked Similarity, 86.6% |
| MAC-SQL | COLING | 2025 | 三 Agent 协作 |
| CHESS | ICML WS | 2025 | Hierarchical Retrieval + Schema Pruning |
| CodeS | SIGMOD | 2024 | SQL-centric 增量预训练 |
| SQL-PaLM | arXiv | 2023 | EXEC-FT 微调 |
| DTS-SQL | EMNLP | 2024 | 两步分解 + 小模型 |
| XiYan-SQL | arXiv | 2025 | 多生成器集成, 89.65% |
| SQLDriller | SIGMOD | 2024 | Execution Consistency 数据清洗 |
| MAGIC | AAAI | 2025 | 自修正 Guideline 自动生成 |
| OpenSearch-SQL | arXiv | 2025 | Consistency Alignment + Voting |
| AutoLink | AAAI | 2025 | Agent 迭代 Schema 探索 |
| RESDSQL | AAAI | 2023 | Cross-encoder + 骨架解耦 |
| X-SQL | arXiv | 2025 | X-Linking SFT, 开源 SOTA |
