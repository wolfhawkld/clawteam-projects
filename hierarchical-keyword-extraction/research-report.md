# 分层关键词抽取与扩展方法研究报告

## 目录
1. [方法综述](#方法综述)
2. [技术架构设计](#技术架构设计)
3. [相关工具/资源列表](#相关工具资源列表)
4. [实现建议](#实现建议)
5. [参考文献](#参考文献)

---

## 方法综述

### 1. 语言学/句式分析的关键词抽取方法

#### 1.1 依存句法分析（Dependency Parsing）

依存句法分析通过识别句子中词语之间的语法依赖关系，帮助理解句子结构并提取关键成分。

**核心概念：**
- **依存关系类型**：主谓关系(nsubj)、动宾关系(obj)、定中关系(amod)等
- **核心词识别**：ROOT节点通常是句子的核心动词或谓语
- **语义焦点**：依存树中权重高的节点往往承载核心语义

**应用方式：**
```python
# 示例：使用spaCy进行依存句法分析
import spacy
nlp = spacy.load("zh_core_web_lg")
doc = nlp("智能手机的市场份额正在快速增长")
for token in doc:
    print(f"{token.text} -> {token.dep_} -> {token.head.text}")
```

**SOTA方法：**
- **Stanza** (Stanford NLP)：多语言支持，基于深度神经网络
- **spaCy**：工业级性能，支持中文（zh_core_web系列模型）
- **HanLP**：中文专用，支持多种依存标注体系
- **LALParser**：联合学习句法和语义依存关系

#### 1.2 语义角色标注（Semantic Role Labeling, SRL）

语义角色标注识别句子中的谓词-论元结构，标注语义角色如施事(Agent)、受事(Patient)、主题(Topic)等。

**核心框架：**
- **FrameNet**：基于框架语义学，定义语义框架和框架元素
- **PropBank**：基于谓词的语义角色标注体系
- **VerbNet**：动词分类和语义角色体系

**最新进展（2024-2025）：**
根据最新综述论文《Semantic Role Labeling: A Systematical Survey》（2025），SRL可有效指导关键语义成分提取，过滤噪声元素如标点和话语填充词。

**关键技术：**
- **TakeFive**：将文本转换为框架导向的知识图谱，结合依存句法和框架识别
- **联合学习**：将依存句法和SRL作为单一任务学习，提高效率
- **神经序列模型**：基于BERT/RoBERTa的端到端SRL系统

#### 1.3 词性标注与名词短语识别

**词性标注（POS Tagging）：**
- 核心词通常为名词(NN)、动词(VV)、形容词(JJ)
- 过滤策略：保留名词、动词、形容词，过滤停用词和功能词

**名词短语识别：**
- **Chunking**：识别NP（名词短语）块
- **依存路径分析**：通过amod、compound等关系识别名词短语边界
- **最长名词短语**：识别句子中完整的概念表达

**中文特性：**
- 中文没有显式空格分隔，需先进行分词
- 量词短语、方位词短语等特殊结构需特殊处理

#### 1.4 信息抽取技术

**关键信息抽取方法：**

| 方法类型 | 代表技术 | 特点 |
|---------|---------|------|
| 统计方法 | TF-IDF, RAKE, YAKE | 无监督，基于词频/位置统计 |
| 图方法 | TextRank, SingleRank | 构建词语共现图，PageRank思想 |
| 嵌入方法 | KeyBERT, KeyLLM | 基于BERT/LLM嵌入，语义匹配 |
| 序列标注 | BERT-Tagger | Token分类任务，端到端抽取 |

**SOTA工具对比：**

| 工具 | 方法 | 优势 | 适用场景 |
|-----|------|------|---------|
| **KeyBERT** | BERT嵌入+相似度 | 语义准确性高 | 语义关键词抽取 |
| **KeyLLM** | LLM生成 | 可生成未见关键词 | 创意性关键词 |
| **YAKE** | 统计特征 | 无需训练，速度快 | 单文档关键词 |
| **RAKE** | 词频+共现 | 简单高效 | 快速提取短语 |
| **TextRank** | 图算法 | 无监督 | 多语言通用 |

---

### 2. 上下位关系扩展方法

#### 2.1 WordNet本体资源

**WordNet结构：**
- **Synset（同义词集）**：表示一个概念的词义集合
- **Hypernym（上位词）**：更抽象的概念层次
- **Hyponym（下位词）**：更具体的概念实例
- **Holonym/Meronym**：部分-整体关系

**扩展策略：**
```
原始关键词: "智能手机"
↓ 上位扩展
- 手机 (hypernym)
- 通信设备 (hypernym)
- 电子设备 (hypernym)
↓ 下位扩展
- iPhone (hyponym)
- Android手机 (hyponym)
- 华为手机 (hyponym)
```

**实现方式：**
```python
import nltk
from nltk.corpus import wordnet

def get_hypernyms(word):
    synsets = wordnet.synsets(word)
    hypernyms = []
    for syn in synsets:
        for hyper in syn.hypernyms():
            hypernyms.extend(hyper.lemma_names())
    return hypernyms
```

#### 2.2 HowNet/E-HowNet中文本体

**HowNet特点：**
- 中文专用语义知识库
- 定义概念的语义特征（义原）
- 提供丰富的语义关系（上下位、同义、反义等）
- **E-HowNet**：扩展版，增强概念表示和关系定义

**核心优势（相比WordNet）：**
- 提供除上下位关系外的更多语义信息
- 通过义原关系编码概念间的语义差异
- 形态-语义结构信息编码

**应用方式：**
- 使用义原距离计算语义相似度
- 通过定义结构提取上下位关系
- 支持跨语言语义扩展

#### 2.3 知识图谱中的概念层级

**主流知识图谱：**

| KG名称 | 语言 | 特点 | 适用场景 |
|-------|------|------|---------|
| **DBpedia** | 多语言 | Wikipedia结构化数据 | 实体概念扩展 |
| **Wikidata** | 多语言 | 综合性知识图谱 | 多语言概念映射 |
| **Freebase** | 英文 | Google知识图谱基础 | 实体关系查询 |
| **CN-DBpedia** | 中文 | 中文百科知识图谱 | 中文实体扩展 |
| **XLore** | 中文 | 中文大规模本体 | 学术概念扩展 |

**概念层级提取：**
```sparql
# SPARQL查询提取上位概念
SELECT ?hypernym WHERE {
  ?concept rdfs:label "智能手机" .
  ?concept skos:broader ?hypernym .
  ?hypernym rdfs:label ?label .
}
```

**实体链接（Entity Linking）：**
将文本中的实体提及链接到知识图谱中的实体，从而获取其概念层级信息。

关键步骤：
1. **候选实体生成**：基于表面形式匹配
2. **实体消歧**：基于上下文和图结构特征
3. **概念扩展**：遍历KG中的概念层级路径

#### 2.4 预训练语言模型中的语义关系

**基于BERT的上下位关系提取：**
- 使用掩码预测识别概念关系
- 通过嵌入相似度发现语义近邻
- **Pattern-based**：识别"X是一种Y"等句式模式

**LLM-based方法：**
- **Prompt-based extraction**：通过精心设计的提示词引导LLM生成扩展关键词
- **Chain-of-thought**：逐步推理概念层级关系
- **Few-shot learning**：少量示例引导扩展方向

**示例Prompt设计：**
```
给定关键词"智能手机"，请从以下角度生成相关关键词：
1. 上位概念（更抽象的类别）
2. 下位概念（具体实例/品牌）
3. 相关概念（功能、属性）

请以JSON格式输出：
{
  "hypernyms": [...],
  "hyponyms": [...],
  "related": [...]
}
```

#### 2.5 统计共现关系

**基于语料库的扩展：**
- **共现统计**：计算词共现频率和分布
- **分布语义**：Word2Vec/GloVe等嵌入模型
- **关联规则**：从大规模文本中挖掘词语关联模式

**TF-IDF扩展：**
结合WordNet同义词和上位词，将语义信息融入TF-IDF权重计算，增强主题提取的语义敏感度。

---

### 3. 分层关键词生成策略

#### 3.1 三层架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    输入：原始查询语句                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 第一层：核心关键词抽取                                         │
│ 方法：依存句法 + 语义角色 + 词性过滤 + KeyBERT                  │
│ 输出：["智能手机", "市场份额", "增长"]                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 第二层：上位扩展（泛化层）                                      │
│ 方法：WordNet/HowNet + KG遍历 + LLM Prompt                     │
│ 输出：["手机", "通信设备", "电子产品", "市场分析", "商业趋势"]     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 第三层：下位扩展（具体化层）                                    │
│ 方法：KG实例查询 + 文本共现 + LLM生成                           │
│ 输出：["iPhone", "华为", "小米", "OPPO", "销量数据", "季度报告"]   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 输出：分层关键词列表（用于检索系统）                             │
│ 格式：{L1: [...], L2: [...], L3: [...]}                       │
└─────────────────────────────────────────────────────────────┘
```

#### 3.2 各层详细策略

**第一层（核心关键词）设计原则：**
- 保留原始查询的核心语义
- 使用语言学分析提取关键名词短语
- 结合语义角色标注识别论元
- 应用KeyBERT/KeyLLM进行语义验证

**第二层（上位扩展）设计原则：**
- 扩展覆盖更广的概念空间
- 保持语义连贯性（避免过度泛化）
- 使用本体资源确保语义准确性
- 控制扩展深度（通常1-3层）

**第三层（下位扩展）设计原则：**
- 提供具体实例和细节
- 增强检索的召回率
- 结合领域知识获取专业术语
- 考虑时序性和地域性因素

#### 3.3 Query Expansion技术

**现代Query Expansion分类：**

| 类型 | 方法 | 特点 |
|------|------|------|
| **Global Analysis** | WordNet/Thesaurus | 基于外部知识库 |
| **Local Analysis** | Pseudo-Relevance Feedback | 基于检索结果扩展 |
| **LLM-based** | Prompt Expansion | 语义生成扩展 |
| **Hybrid** | KG + Statistics + LLM | 多方法融合 |

**LLM-based Query Expansion最新进展：**
- **CSQE (Corpus-Steered QE)**：结合BM25初检索关键句，融合LLM扩展
- **AQE (Aligned Query Expansion)**：通过RSFT/DPO训练，最大化检索效果
- **HyDE**：生成假设性回答文档作为扩展查询

---

## 技术架构设计

### 系统整体架构

```
┌────────────────────────────────────────────────────────────────────┐
│                         分层关键词扩展系统                            │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
│  │  输入模块   │───→│  分析模块   │───→│  扩展模块   │              │
│  └─────────────┘    └─────────────┘    └─────────────┘              │
│       │                  │                  │                       │
│       │                  │                  │                       │
│  ┌────┴────┐        ┌────┴────┐        ┌────┴────┐                  │
│  │文本预处理│        │语言学分析│        │语义扩展 │                  │
│  │- 分词    │        │- 依存句法│        │- 本体查询│                  │
│  │- 去噪    │        │- SRL     │        │- KG遍历 │                  │
│  │- 命名实体│        │- 词性标注│        │- LLM生成│                  │
│  └──────────┘        └──────────┘        └──────────┘                  │
│                           │                  │                       │
│                           └──────────────────┘                       │
│                                    ↓                                 │
│                            ┌─────────────┐                           │
│                            │  融合模块   │                           │
│                            │ - 层级整合  │                           │
│                            │ - 去重去噪  │                           │
│                            │ - 权重计算  │                           │
│                            └─────────────┘                           │
│                                    ↓                                 │
│                            ┌─────────────┐                           │
│                            │  输出模块   │                           │
│                            │ - 结构化输出│                           │
│                            │ - 检索接口  │                           │
│                            └─────────────┘                           │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### 模块详细设计

#### 1. 输入预处理模块

```python
class Preprocessor:
    """文本预处理"""
    
    def __init__(self, language='zh'):
        self.language = language
        if language == 'zh':
            self.segmenter = HanLP()  # 或 spaCy zh_core_web
        else:
            self.nlp = spacy.load('en_core_web_lg')
    
    def process(self, text):
        # 分词
        tokens = self.segment(text)
        # 去除停用词、标点
        tokens = self.filter_stopwords(tokens)
        # 命名实体识别
        entities = self.extract_entities(text)
        return {
            'tokens': tokens,
            'entities': entities,
            'cleaned_text': self.clean_text(text)
        }
```

#### 2. 语言学分析模块

```python
class LinguisticAnalyzer:
    """语言学分析"""
    
    def __init__(self):
        self.nlp = spacy.load('zh_core_web_trf')  # Transformer模型
    
    def analyze(self, text):
        doc = self.nlp(text)
        
        # 依存句法分析
        dependencies = self.extract_dependencies(doc)
        
        # 词性标注与名词短语
        pos_tags = [(t.text, t.pos_) for t in doc]
        noun_phrases = self.extract_noun_phrases(doc)
        
        # 语义角色标注（需要额外模型）
        srl_results = self.semantic_role_labeling(text)
        
        # 核心词识别
        keywords_l1 = self.identify_core_keywords(doc, srl_results)
        
        return {
            'dependencies': dependencies,
            'pos_tags': pos_tags,
            'noun_phrases': noun_phrases,
            'srl': srl_results,
            'keywords_l1': keywords_l1
        }
    
    def identify_core_keywords(self, doc, srl):
        """识别核心关键词"""
        keywords = []
        
        # 策略1：依存树中心词
        root = [t for t in doc if t.dep_ == 'ROOT'][0]
        keywords.append(root.text)
        
        # 策略2：名词短语核心
        for phrase in doc.noun_chunks:
            keywords.append(phrase.text)
        
        # 策略3：语义角色论元
        for role in srl:
            if role['role'] in ['Agent', 'Patient', 'Topic']:
                keywords.append(role['text'])
        
        # 策略4：KeyBERT验证
        kb = KeyBERT()
        kb_keywords = kb.extract_keywords(doc.text)
        keywords.extend([k[0] for k in kb_keywords])
        
        return list(set(keywords))
```

#### 3. 语义扩展模块

```python
class SemanticExpander:
    """语义扩展"""
    
    def __init__(self):
        self.wordnet = self.load_wordnet()
        self.hownet = self.load_hownet()
        self.kg_client = KGClient()  # 知识图谱客户端
        self.llm = LLMClient()       # LLM客户端
    
    def expand_hypernyms(self, keyword, level=2):
        """上位扩展"""
        hypernyms = []
        
        # 方法1：本体查询
        wn_hyper = self.wordnet.get_hypernyms(keyword)
        hn_hyper = self.hownet.get_hypernyms(keyword)
        
        # 方法2：知识图谱遍历
        kg_hyper = self.kg_client.query_broader(keyword)
        
        # 方法3：LLM生成
        llm_hyper = self.llm.generate_hypernyms(keyword)
        
        # 融合结果
        hypernyms = self.merge_and_rank(
            wn_hyper, hn_hyper, kg_hyper, llm_hyper
        )
        
        # 递归扩展到指定层级
        if level > 1:
            for h in hypernyms[:5]:  # 控制扩展广度
                sub_hyper = self.expand_hypernyms(h, level-1)
                hypernyms.extend(sub_hyper)
        
        return hypernyms
    
    def expand_hyponyms(self, keyword, level=2):
        """下位扩展"""
        hyponyms = []
        
        # 方法1：本体查询
        wn_hypo = self.wordnet.get_hyponyms(keyword)
        hn_hypo = self.hownet.get_hyponyms(keyword)
        
        # 方法2：知识图谱实例
        kg_hypo = self.kg_client.query_narrower(keyword)
        
        # 方法3：语料共现（获取高频实例）
        corpus_hypo = self.get_corpus_instances(keyword)
        
        # 方法4：LLM生成具体实例
        llm_hypo = self.llm.generate_hyponyms(keyword)
        
        hyponyms = self.merge_and_rank(
            wn_hypo, hn_hypo, kg_hypo, corpus_hypo, llm_hypo
        )
        
        return hyponyms
    
    def generate_keywords_hierarchy(self, keywords_l1):
        """生成完整三层关键词结构"""
        result = {
            'L1': keywords_l1,  # 核心层
            'L2': [],           # 上位层
            'L3': []            # 下位层
        }
        
        for kw in keywords_l1:
            # 上位扩展
            hypernyms = self.expand_hypernyms(kw, level=2)
            result['L2'].extend(hypernyms)
            
            # 下位扩展
            hyponyms = self.expand_hyponyms(kw, level=2)
            result['L3'].extend(hyponyms)
        
        # 去重与权重排序
        result['L2'] = self.dedupe_and_rank(result['L2'])
        result['L3'] = self.dedupe_and_rank(result['L3'])
        
        return result
```

#### 4. 检索融合模块

```python
class RetrievalFusion:
    """检索融合"""
    
    def search_with_hierarchy(self, query, hierarchy, retriever):
        """使用分层关键词进行检索"""
        all_results = []
        
        # 第一层：精确匹配
        l1_results = retriever.search(hierarchy['L1'], mode='exact')
        all_results.extend(l1_results)
        
        # 第二层：语义扩展（召回）
        l2_results = retriever.search(hierarchy['L2'], mode='semantic')
        all_results.extend(l2_results)
        
        # 第三层：实例补充
        l3_results = retriever.search(hierarchy['L3'], mode='keyword')
        all_results.extend(l3_results)
        
        # 融合与排序
        merged = self.merge_results(all_results)
        ranked = self.rank_results(merged, query, hierarchy)
        
        return ranked
```

---

## 相关工具资源列表

### 1. 中文NLP工具

| 工具 | 类型 | 特点 | GitHub/官网 |
|-----|------|------|------------|
| **HanLP** | 综合NLP | 分词、依存句法、NER等 | https://github.com/hankcs/HanLP |
| **spaCy (zh)** | 综合NLP | 工业级，Transformer模型 | https://spacy.io/models/zh |
| **Jieba** | 分词 | 简单易用，5年未更新（考虑替代） | https://github.com/fxsjy/jieba |
| **LAC** | 百度NLP | 分词+词性+NER联合分析 | https://github.com/baidu/lac |
| **pkuseg** | 北大分词 | 多领域分词 | https://github.com/lancopku/pkuseg |

### 2. 关键词抽取工具

| 工具 | 方法 | 语言支持 | GitHub |
|-----|------|---------|--------|
| **KeyBERT** | BERT嵌入 | 多语言 | https://github.com/MaartenGr/KeyBERT |
| **KeyLLM** | LLM生成 | 多语言 | KeyBERT扩展 |
| **YAKE** | 统计 | 多语言 | https://github.com/LIAAD/yake |
| **RAKE** | 共现 | 多语言 | https://github.com/aneesha/RAKE |
| **TextRank** | 图算法 | 多语言 | https://github.com/davidadamojr/TextRank |
| **jieba.analyse** | TF-IDF/TextRank | 中文 | Jieba内置 |

### 3. 本体与知识图谱资源

| 资源 | 类型 | 语言 | 获取方式 |
|-----|------|------|---------|
| **WordNet** | 英语本体 | 英文 | NLTK内置 |
| **HowNet** | 中文本体 | 中文 | http://www.keenage.com |
| **E-HowNet** | 扩展本体 | 中文 | 台湾中研院CKIP |
| **Chinese WordNet** | 中文WordNet | 中文 | 台湾中研院 |
| **DBpedia** | 通用KG | 多语言 | https://dbpedia.org |
| **Wikidata** | 通用KG | 多语言 | https://wikidata.org |
| **CN-DBpedia** | 中文KG | 中文 | https://github.com/thunlp/CN-DBpedia |
| **OwnThink** | 中文KG | 中文 | https://ownthink.com |

### 4. 语义角色标注工具

| 工具 | 语言支持 | 特点 |
|-----|---------|------|
| **Stanza** | 多语言 | Stanford NLP，端到端SRL |
| **AllenNLP** | 英文 | 基于BERT的SRL模型 |
| **LTP** | 中文 | 哈工大语言技术平台 |
| **simpleSRL** | 英文 | 轻量级SRL |

### 5. LLM工具与框架

| 工具/框架 | 特点 | 用途 |
|----------|------|------|
| **OpenAI API** | GPT系列 | Query扩展、关键词生成 |
| **Claude API** | Anthropic | 语义推理、概念扩展 |
| **LangChain** | LLM应用框架 | 构建扩展Pipeline |
| **Haystack** | RAG框架 | Query Expansion组件 |
| **LlamaIndex** | RAG框架 | 知识图谱集成 |

### 6. Benchmark数据集

| 数据集 | 任务 | 语言 |
|-------|------|------|
| **SemEval-2010** | Keyphrase Extraction | 英文 |
| **Inspec** | Keyphrase Extraction | 英文 |
| **KDD** | Keyphrase Extraction | 英文 |
| **KP20k** | Keyphrase Generation | 英文 |
| **PubMed** | Keyphrase Extraction | 英文医学 |
| **CTB** | 中文句法分析 | 中文 |
| **CoNLL-2009** | SRL | 多语言 |

---

## 实现建议

### 1. 分阶段实施路线

**Phase 1：基础能力搭建（1-2周）**
- 实现文本预处理模块（分词、实体识别）
- 集成KeyBERT进行基础关键词抽取
- 构建WordNet/HowNet查询接口

**Phase 2：语言学分析增强（2-3周）**
- 集成依存句法分析（spaCy/HanLP）
- 实现名词短语识别和核心词提取
- 添加语义角色标注（可选，视需求复杂度）

**Phase 3：语义扩展实现（2-3周）**
- 实现本体上下位查询
- 集成知识图谱API
- 设计LLM Prompt进行语义扩展

**Phase 4：融合与优化（1-2周）**
- 实现三层关键词融合逻辑
- 设计权重计算和排序策略
- 与检索系统集成测试

### 2. 技术选型建议

**中文处理推荐方案：**
```
预处理: HanLP v2 (最新版) 或 spaCy zh_core_web_trf
关键词: KeyBERT + paraphrase-multilingual-MiniLM-L12-v2
本体:   HowNet (中文语义) + CN-DBpedia (实体)
扩展:   LLM (GPT-4/Claude) + KG查询
```

**英文处理推荐方案：**
```
预处理: spaCy en_core_web_trf
关键词: KeyBERT + paraphrase-MiniLM-L6-v2
本体:   WordNet + DBpedia
扩展:   LLM + KG + 统计共现
```

### 3. LLM Prompt设计模板

```markdown
# 关键词分层扩展Prompt模板

## 输入
原始查询：{query}
核心关键词（已提取）：{keywords_l1}

## 任务
请基于核心关键词，生成分层扩展关键词：

### 上位扩展（L2层）
- 提取更抽象的概念类别
- 保持语义相关性
- 控制扩展层级深度（最多3层）

### 下位扩展（L3层）
- 提取具体实例和品牌
- 包含相关属性和功能
- 考虑领域专业术语

## 输出格式
{
  "original_query": "...",
  "L1_core": ["..."],
  "L2_hypernyms": [
    {"keyword": "...", "level": 1, "source": "本体/LLM"},
    ...
  ],
  "L3_hyponyms": [
    {"keyword": "...", "type": "实例/属性/功能", "source": "..."},
    ...
  ]
}

## 注意事项
- 每层关键词数量控制在10-20个
- 确保语义连贯，避免过度发散
- 标注每个扩展关键词的来源（便于后续评估）
```

### 4. 性能优化建议

**缓存策略：**
- 缓存本体查询结果（WordNet/HowNet）
- 缓存KG实体和概念路径
- 使用向量数据库存储关键词嵌入

**批量处理：**
- 使用异步并发查询KG和LLM
- 批量生成嵌入向量
- 合并相似查询请求

**质量控制：**
- 设置语义相似度阈值（过滤噪声）
- 控制扩展深度（避免过度泛化）
- 保留扩展来源标注（便于溯源）

### 5. 评估方法

**关键词质量评估：**
- **Precision**：扩展关键词与原始语义的相关性
- **Coverage**：扩展关键词覆盖概念空间的广度
- **Ranking Accuracy**：关键词排序与人工标注的一致性

**检索效果评估：**
- **Recall@k**：使用扩展关键词的召回率提升
- **nDCG**：检索结果排序质量
- **MAP**：平均检索精度

---

## 参考文献

### 学术论文

1. **Semantic Role Labeling综述** (2025)
   - 《Semantic Role Labeling: A Systematical Survey》
   - arXiv:2502.08660

2. **TakeFive - 知识图谱抽取** (2018)
   - 《Semantic Role Labeling for Knowledge Graph Extraction from Text》
   - arXiv:1811.01409

3. **上下位关系提取** (2024)
   - 《Automatic Hypernym-Hyponym Relation Extraction With WordNet》
   - Semantic Scholar: ab2052d0645f651f5ec1b13b82b35f505349c2f9

4. **KeyBERT关键词抽取** (2020)
   - Maarten Grootendorst
   - https://github.com/MaartenGr/KeyBERT

5. **LLM-TAKE关键词生成** (2024)
   - 《LLM-TAKE: Theme-Aware Keyword Extraction Using Large Language Models》
   - arXiv:2312.0909

6. **Query Expansion综述** (2025)
   - 《Semantic approaches for query expansion: taxonomy, challenges》
   - PMC: PMC11935759

7. **E-HowNet** (2009)
   - 《E-HowNet: A Lexical Semantic Representation System》
   - ACL Anthology: Y09-1001

8. **知识图谱实体检索** (2018)
   - 《Entity Retrieval in the Knowledge Graph with Hierarchical Types》
   - ACM: 10.1145/3234944.3234963

### 工具文档

1. **spaCy Documentation**
   - https://spacy.io/models/zh

2. **HanLP Documentation**
   - https://hanlp.hankcs.com

3. **KeyBERT Guide**
   - https://maartengr.github.io/KeyBERT

4. **Haystack Query Expansion**
   - https://haystack.deepset.ai/blog/query-expansion

### 在线资源

1. **HowNet官网**
   - http://www.keenage.com/html/index.html

2. **E-HowNet**
   - https://ckip.iis.sinica.edu.tw

3. **WordNet (NLTK)**
   - https://www.nltk.org/howto/wordnet.html

---

## 附录：示例实现代码

### 完整Pipeline示例

```python
#!/usr/bin/env python3
"""
分层关键词抽取与扩展系统示例实现
"""

from typing import List, Dict
import spacy
from keybert import KeyBERT
import nltk
from nltk.corpus import wordnet as wn

class HierarchicalKeywordExtractor:
    def __init__(self, lang='zh'):
        # 初始化NLP组件
        self.nlp = spacy.load('zh_core_web_lg')
        self.kb = KeyBERT('paraphrase-multilingual-MiniLM-L12-v2')
        nltk.download('wordnet')
        
    def extract(self, text: str) -> Dict[str, List[str]]:
        """完整抽取流程"""
        # Step 1: 预处理
        doc = self.nlp(text)
        
        # Step 2: L1 - 核心关键词
        l1_keywords = self._extract_l1(doc, text)
        
        # Step 3: L2 - 上位扩展
        l2_keywords = self._expand_hypernyms(l1_keywords)
        
        # Step 4: L3 - 下位扩展  
        l3_keywords = self._expand_hyponyms(l1_keywords)
        
        return {
            'L1': l1_keywords,
            'L2': l2_keywords,
            'L3': l3_keywords
        }
    
    def _extract_l1(self, doc, text) -> List[str]:
        """提取核心关键词"""
        keywords = []
        
        # 依存树核心词
        for token in doc:
            if token.dep_ == 'ROOT':
                keywords.append(token.text)
        
        # 名词短语
        for chunk in doc.noun_chunks:
            keywords.append(chunk.text)
        
        # KeyBERT补充
        kb_keywords = self.kb.extract_keywords(text, top_n=5)
        keywords.extend([k[0] for k in kb_keywords])
        
        return list(set(keywords))
    
    def _expand_hypernyms(self, keywords: List[str]) -> List[str]:
        """上位扩展"""
        expanded = []
        for kw in keywords:
            # WordNet上位词
            synsets = wn.synsets(kw, lang='cmn')  # 中文WordNet
            for syn in synsets:
                for hyper in syn.hypernyms():
                    expanded.extend(hyper.lemma_names('cmn'))
        return list(set(expanded))
    
    def _expand_hyponyms(self, keywords: List[str]) -> List[str]:
        """下位扩展"""
        expanded = []
        for kw in keywords:
            synsets = wn.synsets(kw, lang='cmn')
            for syn in synsets:
                for hypo in syn.hyponyms():
                    expanded.extend(hypo.lemma_names('cmn'))
        return list(set(expanded))

# 使用示例
if __name__ == '__main__':
    extractor = HierarchicalKeywordExtractor()
    result = extractor.extract("智能手机的市场份额正在快速增长")
    print(f"L1 (核心): {result['L1']}")
    print(f"L2 (上位): {result['L2']}")
    print(f"L3 (下位): {result['L3']}")
```

---

*报告生成时间：2026-04-10*
*版本：v1.0*