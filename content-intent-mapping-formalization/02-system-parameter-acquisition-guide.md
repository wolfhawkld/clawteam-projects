# 系统参数获取方式定义 - 完整实现指南

> **版本**: v1.0  
> **创建时间**: 2026-04-03  
> **团队**: content-intent-math  
> **作者**: agent-system  
> **任务**: 系统参数获取方式定义

---

## 执行摘要

本指南定义内容-意图映射系统的核心参数获取方式，涵盖四个关键维度：

| 维度 | 核心参数 | 获取方式 | 验证目标 |
|------|---------|---------|---------|
| **嵌入向量** | φ(c), ψ(i) | 预训练模型 + 领域适配 | 语义保真度 > 0.85 |
| **反馈信号** | r(u,c,i) | 显式/隐式/CoT融合量化 | 信号有效比 > 70% |
| **置信度** | P(i|c) | LLM推断 + 贝叶斯校准 | 校准误差 < 0.05 |
| **学习率** | η(t) | 自适应调度 + 阶段策略 | 收敛速度 < 100 epochs |

---

## 一、嵌入向量获取

### 1.1 数学定义

内容-意图映射系统的核心嵌入函数：

$$\phi(c): \mathcal{C} \rightarrow \mathbb{R}^{d_c}$$ — Chunk内容编码器  
$$\psi(i): \mathcal{I} \rightarrow \mathbb{R}^{d_i}$$ — Intent意图编码器  

映射关系：
$$s(c, i) = \langle \phi(c), \psi(i) \rangle$$ — 内容-意图匹配得分

### 1.2 Chunk编码器 φ(c) 选择与配置

#### 1.2.1 模型选择矩阵

| 模型 | 维度 | 中英文支持 | 延迟 | 推荐场景 |
|------|------|----------|------|---------|
| **bge-m3** | 1024 | ✅ 优秀 | 15-20ms | 🏆 企业应用首选 |
| **text-embedding-3-large** | 3072 | ✅ 良好 | 30-50ms | 高精度场景 |
| **text-embedding-3-small** | 1536 | ✅ 良好 | 15-20ms | 成本敏感 |
| **paraphrase-multilingual-mpnet-base-v2** | 768 | ✅ 良好 | 10-15ms | 边缘部署 |
| **all-MiniLM-L6-v2** | 384 | ❌ 英文优先 | 5-10ms | 极速场景 |

#### 1.2.2 推荐配置

```yaml
# chunk_encoder_config.yaml

chunk_encoder:
  model: "bge-m3"  # 推荐：中英文平衡，性价比高
  
  # 嵌入配置
  embedding:
    dimension: 1024           # 输出维度
    normalize: true           # L2归一化（推荐开启）
    batch_size: 32            # 批处理大小
    
  # Chunk处理
  chunking:
    max_length: 512           # 最大token长度
    overlap: 50               # Chunk重叠（避免语义断裂）
    strategy: "semantic"      # 语义分块优于固定长度
    
  # 性能优化
  optimization:
    quantization: "int8"      # INT8量化（延迟降低30-50%）
    cache_enabled: true       # 启用嵌入缓存
    cache_ttl: 3600           # 缓存有效期（秒）
```

#### 1.2.3 Chunk预处理流程

```python
# chunk_preprocessing.py

class ChunkProcessor:
    """Chunk预处理与编码"""
    
    def __init__(self, encoder_model: str = "bge-m3"):
        self.encoder = SentenceTransformer(encoder_model)
        self.max_length = 512
        self.overlap = 50
    
    def process_content(self, content: str) -> list[ChunkEmbedding]:
        """处理内容并生成嵌入向量"""
        
        # 1. 语义分块
        chunks = self._semantic_chunk(content)
        
        # 2. 批量编码
        embeddings = self.encoder.encode(
            chunks,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False
        )
        
        # 3. 构建结果
        results = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            results.append(ChunkEmbedding(
                chunk_id=f"chunk_{idx}",
                content=chunk,
                embedding=embedding,  # φ(c) ∈ ℝ^1024
                metadata={
                    'position': idx,
                    'length': len(chunk),
                    'encoder': self.encoder_name
                }
            ))
        
        return results
    
    def _semantic_chunk(self, content: str) -> list[str]:
        """语义分块（优于固定长度）"""
        # 使用语义边界检测
        semantic_boundaries = self._detect_semantic_boundaries(content)
        
        chunks = []
        for start, end in semantic_boundaries:
            chunk = content[start:end]
            if len(chunk) > self.max_length:
                # 超长Chunk二次分割
                sub_chunks = self._recursive_split(chunk)
                chunks.extend(sub_chunks)
            else:
                chunks.append(chunk)
        
        return chunks
```

### 1.3 Intent编码器 ψ(i) 选择与配置

#### 1.3.1 模型选择矩阵

Intent编码需要更强的语义理解能力，推荐使用对比学习优化的模型：

| 模型 | 特点 | 推荐度 | 适用意图类型 |
|------|------|-------|-------------|
| **bge-m3** | 多语言对比学习优化 | 🏆 首选 | 全意图类型 |
| **SimCSE-roberta-large** | 自监督对比学习 | ★★★★ | 语义相似意图 |
| **DeCL-tuned** | 聚类+对比联合优化 | ★★★★ | 新意图发现 |
| **text-embedding-3-large** | 高维度表达 | ★★★ | 精细意图 |

#### 1.3.2 Intent嵌入与意图Schema绑定

```python
# intent_encoder.py

class IntentEncoder:
    """意图编码器，与意图Schema绑定"""
    
    def __init__(self, encoder_model: str = "bge-m3"):
        self.encoder = SentenceTransformer(encoder_model)
        self.intent_schema = self._load_intent_schema()
        self.intent_prototypes = {}  # ψ(i) 原型向量
        
    def initialize_prototypes(self, seed_data: dict = None):
        """初始化意图原型向量"""
        
        # 方案A：零样本启动（使用Speech Act 5类）
        if seed_data is None:
            return self._zero_shot_initialize()
        
        # 方案B：种子数据启动
        return self._seed_data_initialize(seed_data)
    
    def _zero_shot_initialize(self):
        """零样本：基于Speech Act定义生成原型"""
        SPEECH_ACT_DEFINITIONS = {
            "L_ASSERTIVE": "陈述事实，让听者相信某事为真。典型表达：X是Y、我发现、根据数据显示",
            "L_DIRECTIVE": "指令行为，希望听者执行某事。典型表达：请帮我、怎么做、如何才能",
            "L_COMMISSIVE": "承诺行为，说话者承诺未来行动。典型表达：我会、保证、承诺",
            "L_EXPRESSIVE": "表达行为，表达心理状态。典型表达：谢谢、抱歉、太棒了",
            "L_DECLARATIVE": "宣告行为，通过言语改变世界状态。典型表达：我宣布、任命、生效"
        }
        
        # 使用定义文本生成原型嵌入
        for intent_id, definition in SPEECH_ACT_DEFINITIONS.items():
            embedding = self.encoder.encode(definition, normalize_embeddings=True)
            self.intent_prototypes[intent_id] = embedding  # ψ(i)
        
        return self.intent_prototypes
    
    def _seed_data_initialize(self, seed_data: dict):
        """种子数据：从样本计算原型"""
        for intent_id, samples in seed_data.items():
            sample_texts = [s['text'] for s in samples]
            embeddings = self.encoder.encode(sample_texts, normalize_embeddings=True)
            # 原型 = 样本嵌入中心
            prototype = np.mean(embeddings, axis=0)
            prototype = prototype / np.linalg.norm(prototype)  # 归一化
            self.intent_prototypes[intent_id] = prototype
        
        return self.intent_prototypes
    
    def encode_intent(self, intent_query: str) -> np.ndarray:
        """编码用户查询为意图向量 ψ(i)"""
        return self.encoder.encode(intent_query, normalize_embeddings=True)
```

### 1.4 向量维度推荐

#### 1.4.1 维度选择策略

| 场景 | 推荐维度 | 理由 |
|------|---------|------|
| **企业生产** | 768-1024 | 平衡表达力与性能 |
| **高精度研究** | 1536-3072 | 最大语义保真度 |
| **边缘部署** | 384-512 | 延迟优先 |
| **意图聚类** | 256-768 | 聚类效果稳定 |

#### 1.4.2 内容-意图向量空间对齐

关键问题：φ(c) 和 ψ(i) 是否需要在同一向量空间？

```
┌─────────────────────────────────────────────────────────────┐
│           内容-意图向量空间对齐策略                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  方案A: 同空间编码 (推荐)                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  φ(c) 和 ψ(i) 使用同一编码器                         │   │
│  │  优点：直接计算余弦相似度 s(c,i) = ⟨φ(c), ψ(i)⟩      │   │
│  │  适用：意图分类、相似度检索                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  方案B: 跨空间编码 + 映射层                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  φ(c) ∈ ℝ^d_c, ψ(i) ∈ ℝ^d_i                         │   │
│  │  需要映射层 M: ℝ^d_c → ℝ^d_i                         │   │
│  │  s(c,i) = ⟨M·φ(c), ψ(i)⟩                            │   │
│  │  适用：异构模型组合、领域迁移                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**推荐方案A**：使用同一编码器（如 bge-m3），简化系统架构。

#### 1.4.3 维度压缩可选

当维度过高影响性能时，可使用降维：

```python
# dimension_reduction.py

class VectorCompressor:
    """向量维度压缩"""
    
    def __init__(self, target_dim: int = 256):
        self.target_dim = target_dim
        self.pca = None
    
    def fit(self, embeddings: np.ndarray):
        """训练PCA降维"""
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=self.target_dim)
        self.pca.fit(embeddings)
    
    def compress(self, embedding: np.ndarray) -> np.ndarray:
        """压缩向量"""
        compressed = self.pca.transform(embedding.reshape(1, -1))
        return compressed.flatten()
    
    def evaluate_quality(self, original: np.ndarray, compressed: np.ndarray):
        """评估压缩质量"""
        # 计算语义保真度
        fidelity = cosine_similarity(original, self.pca.inverse_transform(compressed.reshape(1, -1)))
        return fidelity
```

---

## 二、反馈信号量化

### 2.1 数学定义

反馈信号函数：
$$r(u, c, i) \in \mathbb{R}$$ — 用户u对内容c-意图i匹配的反馈

反馈类型：
- $r_{explicit}$ — 显式反馈（点赞/踩）
- $r_{implicit}$ — 隐式反馈（行为推断）
- $r_{cot}$ — CoT融合信号（推理链路）

总反馈信号：
$$r(u, c, i) = \alpha \cdot r_{explicit} + \beta \cdot r_{implicit} + \gamma \cdot r_{cot}$$

### 2.2 显式反馈量化

#### 2.2.1 数值映射规则

| 显式行为 | 数值 | 权重衰减 |
|---------|------|---------|
| 👍 点赞 | +1.0 | 无 |
| 👎 踩 | -1.0 | 无 |
| ⭐⭐⭐⭐⭐ 5星评分 | +1.0 | 无 |
| ⭐⭐⭐⭐ 4星评分 | +0.8 | 无 |
| ⭐⭐⭐ 3星评分 | +0.5 | 轻微负信号 |
| ⭐⭐ 2星评分 | -0.5 | 明确负信号 |
| ⭐ 1星评分 | -1.0 | 强负信号 |

#### 2.2.2 实现代码

```python
# explicit_feedback.py

from dataclasses import dataclass
from enum import Enum

class ExplicitFeedbackType(Enum):
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    STAR_RATING = "star_rating"
    ACCEPT = "accept"        # 用户采纳回答
    REJECT = "reject"        # 用户明确拒绝
    HELPFUL = "helpful"      # 标记有用
    NOT_HELPFUL = "not_helpful"

@dataclass
class ExplicitFeedback:
    feedback_type: ExplicitFeedbackType
    value: float            # 原始值
    weight: float = 1.0     # 可调权重
    timestamp: float = None

EXPLICIT_FEEDBACK_MAPPING = {
    ExplicitFeedbackType.THUMBS_UP: 1.0,
    ExplicitFeedbackType.THUMBS_DOWN: -1.0,
    ExplicitFeedbackType.STAR_RATING: lambda x: (x - 3) / 2,  # 1-5星映射到 [-1, 1]
    ExplicitFeedbackType.ACCEPT: 1.0,
    ExplicitFeedbackType.REJECT: -1.0,
    ExplicitFeedbackType.HELPFUL: 1.0,
    ExplicitFeedbackType.NOT_HELPFUL: -1.0,
}

class ExplicitFeedbackQuantifier:
    """显式反馈量化器"""
    
    def quantify(self, feedback: ExplicitFeedback) -> float:
        """量化显式反馈为数值信号"""
        mapping = EXPLICIT_FEEDBACK_MAPPING[feedback.feedback_type]
        
        if callable(mapping):
            # 复杂映射（如星级）
            r_explicit = mapping(feedback.value)
        else:
            # 简单映射
            r_explicit = mapping
        
        # 应用权重
        return r_explicit * feedback.weight
```

### 2.3 隐式反馈量化

#### 2.3.1 隐式行为检测方法

| 隐式行为 | 检测方法 | 数值映射 | 说明 |
|---------|---------|---------|------|
| **追问** | 对话轮次检测 | +0.3 | 用户继续探索 = 内容相关 |
| **转话题** | 语义距离突变 | -0.5 | 用户放弃 = 不匹配 |
| **停留时长** | 阅读时间 > 预期 | +0.2 | 深度阅读 = 有价值 |
| **快速退出** | 响应后立即离开 | -0.3 | 无效回答信号 |
| **复制内容** | 复制操作检测 | +0.5 | 用户采纳内容 |
| **分享转发** | 分享行为检测 | +0.8 | 高价值认可 |
| **重试请求** | 相似问题重复 | -0.4 | 回答不满意 |

#### 2.3.2 隐式反馈检测实现

```python
# implicit_feedback.py

from dataclasses import dataclass
import numpy as np

@dataclass
class ImplicitFeedbackSignal:
    signal_type: str
    detected: bool
    intensity: float         # 信号强度
    confidence: float        # 检测置信度
    metadata: dict = None

class ImplicitFeedbackDetector:
    """隐式反馈检测器"""
    
    def __init__(self, encoder_model: str = "bge-m3"):
        self.encoder = SentenceTransformer(encoder_model)
        self.topic_shift_threshold = 0.5  # 语义距离阈值
        self.expected_reading_time = None  # 预期阅读时间
    
    def detect_follow_up(self, conversation_history: list[dict]) -> ImplicitFeedbackSignal:
        """检测追问信号"""
        # 追问特征：后续问题与前一回答语义相关
        if len(conversation_history) < 2:
            return ImplicitFeedbackSignal(signal_type="follow_up", detected=False, intensity=0, confidence=0)
        
        last_query = conversation_history[-1]['query']
        prev_answer = conversation_history[-2]['answer']
        
        # 计算语义相似度
        query_embedding = self.encoder.encode(last_query)
        answer_embedding = self.encoder.encode(prev_answer)
        similarity = cosine_similarity(query_embedding, answer_embedding)
        
        detected = similarity > 0.3  # 语义相关阈值
        intensity = similarity if detected else 0
        
        return ImplicitFeedbackSignal(
            signal_type="follow_up",
            detected=detected,
            intensity=0.3 if detected else 0,  # 追问 = +0.3
            confidence=abs(similarity - 0.3) / 0.7,  # 置信度与距离阈值相关
            metadata={'similarity': similarity}
        )
    
    def detect_topic_shift(self, conversation_history: list[dict]) -> ImplicitFeedbackSignal:
        """检测话题转换信号"""
        if len(conversation_history) < 2:
            return ImplicitFeedbackSignal(signal_type="topic_shift", detected=False, intensity=0, confidence=0)
        
        current_query = conversation_history[-1]['query']
        prev_query = conversation_history[-2]['query']
        
        # 计算语义距离
        current_embedding = self.encoder.encode(current_query)
        prev_embedding = self.encoder.encode(prev_query)
        distance = 1 - cosine_similarity(current_embedding, prev_embedding)
        
        detected = distance > self.topic_shift_threshold
        intensity = -0.5 if detected else 0  # 话题转换 = -0.5
        
        return ImplicitFeedbackSignal(
            signal_type="topic_shift",
            detected=detected,
            intensity=intensity,
            confidence=abs(distance - self.topic_shift_threshold) / 0.5,
            metadata={'semantic_distance': distance}
        )
    
    def detect_engagement(self, engagement_data: dict) -> ImplicitFeedbackSignal:
        """检测用户参与度信号"""
        # 停留时长检测
        reading_time = engagement_data.get('reading_time', 0)
        content_length = engagement_data.get('content_length', 0)
        
        expected_time = self._estimate_reading_time(content_length)
        time_ratio = reading_time / expected_time
        
        if time_ratio > 1.5:
            # 深度阅读
            return ImplicitFeedbackSignal(
                signal_type="deep_reading",
                detected=True,
                intensity=+0.2,
                confidence=min(1.0, (time_ratio - 1.5) / 1.0)
            )
        elif time_ratio < 0.3:
            # 快速退出
            return ImplicitFeedbackSignal(
                signal_type="quick_exit",
                detected=True,
                intensity=-0.3,
                confidence=min(1.0, (0.3 - time_ratio) / 0.3)
            )
        
        return ImplicitFeedbackSignal(signal_type="engagement", detected=False, intensity=0, confidence=0)
    
    def detect_copy_action(self, action_data: dict) -> ImplicitFeedbackSignal:
        """检测复制行为"""
        if action_data.get('copy_detected', False):
            return ImplicitFeedbackSignal(
                signal_type="copy_content",
                detected=True,
                intensity=+0.5,
                confidence=1.0,
                metadata={'copied_text_length': action_data.get('copied_length', 0)}
            )
        return ImplicitFeedbackSignal(signal_type="copy", detected=False, intensity=0, confidence=0)
    
    def _estimate_reading_time(self, content_length: int) -> float:
        """预估阅读时间（中文：300字/分钟）"""
        return content_length / 300 * 60  # 秒
```

#### 2.3.3 隐式反馈聚合

```python
class ImplicitFeedbackAggregator:
    """隐式反馈聚合器"""
    
    IMPLICIT_SIGNAL_WEIGHTS = {
        'follow_up': 0.25,
        'topic_shift': 0.35,
        'deep_reading': 0.15,
        'quick_exit': 0.20,
        'copy_content': 0.30,
    }
    
    def aggregate(self, signals: list[ImplicitFeedbackSignal]) -> float:
        """聚合多个隐式信号"""
        total_signal = 0
        total_weight = 0
        
        for signal in signals:
            if signal.detected:
                weight = self.IMPLICIT_SIGNAL_WEIGHTS.get(signal.signal_type, 0.1)
                weighted_signal = signal.intensity * signal.confidence * weight
                total_signal += weighted_signal
                total_weight += weight
        
        # 归一化
        if total_weight > 0:
            return total_signal / total_weight
        return 0
```

### 2.4 CoT融合信号量化

#### 2.4.1 CoT信号类型

| CoT信号类型 | 检测方法 | 强度估计 | 说明 |
|------------|---------|---------|------|
| **纠偏信号** | 用户纠正推理路径 | -0.5 ~ -1.0 | 系统推理方向错误 |
| **补充信号** | 用户添加信息 | +0.3 ~ +0.5 | 系统推理不完整 |
| **打断信号** | 用户中止输出 | -0.2 ~ -0.5 | 推理无意义/太长 |
| **确认信号** | 用户确认中间步骤 | +0.2 | 推理路径正确 |
| **追问细节** | 用户深挖某步骤 | +0.1 | 推理有吸引力 |

#### 2.4.2 CoT信号提取实现

```python
# cot_feedback.py

from dataclasses import dataclass
from enum import Enum

class CoTSignalType(Enum):
    CORRECTION = "correction"      # 纠偏
    SUPPLEMENT = "supplement"      # 补充
    INTERRUPTION = "interruption"  # 打断
    CONFIRMATION = "confirmation"  # 确认
    DEEP_DIVE = "deep_dive"        # 追问细节

@dataclass
class CoTSignal:
    signal_type: CoTSignalType
    intensity: float
    confidence: float
    position: int           # 在CoT链中的位置
    content: str = None

COT_INTENSITY_ESTIMATION = {
    # 纠偏强度：基于纠正的严重程度
    CoTSignalType.CORRECTION: {
        'minor': -0.5,      # 小修正
        'major': -0.8,      # 大修正
        'fundamental': -1.0  # 根本错误
    },
    # 补充强度：基于补充信息量
    CoTSignalType.SUPPLEMENT: {
        'small': +0.2,
        'medium': +0.3,
        'large': +0.5
    },
    # 其他固定值
    CoTSignalType.INTERRUPTION: -0.3,
    CoTSignalType.CONFIRMATION: +0.2,
    CoTSignalType.DEEP_DIVE: +0.1,
}

class CoTFeedbackExtractor:
    """CoT推理链反馈信号提取器"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def extract_from_interaction(self, cot_steps: list[str], user_response: str) -> list[CoTSignal]:
        """从用户响应中提取CoT反馈信号"""
        
        signals = []
        
        # 1. 检测打断信号
        if self._detect_interruption(user_response):
            signals.append(CoTSignal(
                signal_type=CoTSignalType.INTERRUPTION,
                intensity=-0.3,
                confidence=0.8,
                position=-1  # 全链打断
            ))
        
        # 2. LLM分析用户响应意图
        analysis = self._analyze_response_intent(cot_steps, user_response)
        
        # 3. 解析分析结果
        for item in analysis.get('signals', []):
            signal_type = CoTSignalType(item['type'])
            intensity = self._estimate_intensity(signal_type, item)
            
            signals.append(CoTSignal(
                signal_type=signal_type,
                intensity=intensity,
                confidence=item.get('confidence', 0.7),
                position=item.get('position', -1),
                content=item.get('content', '')
            ))
        
        return signals
    
    def _detect_interruption(self, user_response: str) -> bool:
        """检测打断信号"""
        # 快速检测打断关键词
        interruption_keywords = ["停止", "够了", "不用继续", "等等", "skip"]
        return any(kw in user_response.lower() for kw in interruption_keywords)
    
    def _analyze_response_intent(self, cot_steps: list[str], user_response: str) -> dict:
        """LLM分析用户响应意图"""
        
        COT_ANALYSIS_PROMPT = """
分析用户对CoT推理步骤的响应，提取反馈信号。

CoT推理步骤：
{cot_steps}

用户响应：{user_response}

请判断：
1. 用户是否纠正了某个推理步骤？（纠偏信号）
2. 用户是否补充了遗漏信息？（补充信号）
3. 用户是否确认了某个步骤？（确认信号）
4. 用户是否追问某个步骤细节？（深挖信号）

输出JSON：
{{
  "signals": [
    {{
      "type": "correction|supplement|confirmation|deep_dive",
      "position": 0-N,
      "severity": "minor|major|fundamental",  // 纠偏时
      "amount": "small|medium|large",         // 补充时
      "confidence": 0.0-1.0,
      "content": "..."
    }}
  ]
}}
"""
        
        prompt = COT_ANALYSIS_PROMPT.format(
            cot_steps="\n".join([f"Step {i}: {s}" for i, s in enumerate(cot_steps)]),
            user_response=user_response
        )
        
        response = self.llm.call(prompt, temperature=0.1)
        return parse_json_response(response)
    
    def _estimate_intensity(self, signal_type: CoTSignalType, item: dict) -> float:
        """估计信号强度"""
        intensity_map = COT_INTENSITY_ESTIMATION[signal_type]
        
        if isinstance(intensity_map, dict):
            # 分级强度
            severity = item.get('severity', item.get('amount', 'medium'))
            return intensity_map.get(severity, intensity_map.get('medium', 0))
        else:
            # 固定强度
            return intensity_map
    
    def aggregate_cot_signals(self, signals: list[CoTSignal]) -> float:
        """聚合CoT信号"""
        if not signals:
            return 0
        
        total = sum(s.intensity * s.confidence for s in signals)
        # 按信号数量归一化
        return total / max(len(signals), 1)
```

### 2.5 反馈信号融合

#### 2.5.1 总反馈信号计算

```python
# feedback_fusion.py

class FeedbackFusionEngine:
    """反馈信号融合引擎"""
    
    def __init__(self, weights: dict = None):
        # 默认权重配置
        self.weights = weights or {
            'explicit': 0.5,     # 显式反馈权重最高
            'implicit': 0.3,     # 隐式反馈辅助
            'cot': 0.2,          # CoT信号补充
        }
    
    def fuse(self, 
             explicit_feedback: list[ExplicitFeedback],
             implicit_signals: list[ImplicitFeedbackSignal],
             cot_signals: list[CoTSignal]) -> float:
        """融合三类反馈信号"""
        
        # 1. 量化显式反馈
        explicit_quantifier = ExplicitFeedbackQuantifier()
        r_explicit = sum(explicit_quantifier.quantify(f) for f in explicit_feedback)
        r_explicit = np.clip(r_explicit, -1, 1)  # 截断到 [-1, 1]
        
        # 2. 聚合隐式反馈
        implicit_aggregator = ImplicitFeedbackAggregator()
        r_implicit = implicit_aggregator.aggregate(implicit_signals)
        r_implicit = np.clip(r_implicit, -1, 1)
        
        # 3. 聚合CoT信号
        cot_extractor = CoTFeedbackExtractor(None)
        r_cot = cot_extractor.aggregate_cot_signals(cot_signals)
        r_cot = np.clip(r_cot, -1, 1)
        
        # 4. 加权融合
        r_total = (
            self.weights['explicit'] * r_explicit +
            self.weights['implicit'] * r_implicit +
            self.weights['cot'] * r_cot
        )
        
        return r_total
    
    def update_weights(self, performance_metrics: dict):
        """动态调整权重"""
        # 基于信号有效比调整权重
        signal_effectiveness = performance_metrics.get('signal_effectiveness', {})
        
        for signal_type, effectiveness in signal_effectiveness.items():
            if effectiveness > 0.7:
                self.weights[signal_type] *= 1.1  # 有效信号增加权重
            elif effectiveness < 0.5:
                self.weights[signal_type] *= 0.9  # 无效信号降低权重
        
        # 归一化权重
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
```

---

## 三、置信度初始化

### 3.1 数学定义

置信度函数：
$$P(i|c) \in [0, 1]$$ — 给定内容c，意图为i的概率

初始化置信度：
$$P_0(i|c) = f_{LLM}(c, i; \theta)$$ — LLM零样本推断

### 3.2 LLM零样本置信度推断

#### 3.2.1 推断方法

```python
# confidence_initialization.py

CONFIDENCE_INFERENCE_PROMPT = """
分析用户查询与意图的匹配置信度。

用户查询：{query}
候选意图：
{intent_candidates}

请判断：
1. 每个候选意图与查询的匹配程度
2. 匹配置信度（0.0-1.0）
3. 判断依据

输出JSON：
{{
  "intent_scores": [
    {{
      "intent_id": "...",
      "confidence": 0.0-1.0,
      "reason": "...",
      "keywords_matched": [...],
      "semantic_similarity": 0.0-1.0
    }}
  ],
  "primary_intent": "...",
  "uncertainty_level": "low|medium|high"
}}
"""

class ConfidenceInitializer:
    """置信度初始化器"""
    
    def __init__(self, llm_client, encoder_model: str = "bge-m3"):
        self.llm = llm_client
        self.encoder = SentenceTransformer(encoder_model)
        self.intent_prototypes = None
    
    def initialize_confidence(self, 
                             query: str, 
                             intent_candidates: list[dict]) -> dict:
        """初始化意图置信度"""
        
        # 方法1：LLM语义推断
        llm_confidence = self._llm_inference(query, intent_candidates)
        
        # 方法2：嵌入相似度计算
        embedding_confidence = self._embedding_similarity(query, intent_candidates)
        
        # 方法3：融合置信度
        fused_confidence = self._fuse_confidence(llm_confidence, embedding_confidence)
        
        return {
            'intent_scores': fused_confidence,
            'primary_intent': self._select_primary(fused_confidence),
            'initialization_method': 'hybrid_llm_embedding'
        }
    
    def _llm_inference(self, query: str, intent_candidates: list[dict]) -> list[dict]:
        """LLM语义推断置信度"""
        
        candidates_text = "\n".join([
            f"{i['intent_id']}: {i['name']} - {i['definition']}"
            for i in intent_candidates
        ])
        
        prompt = CONFIDENCE_INFERENCE_PROMPT.format(
            query=query,
            intent_candidates=candidates_text
        )
        
        response = self.llm.call(prompt, temperature=0.1)
        result = parse_json_response(response)
        
        return result.get('intent_scores', [])
    
    def _embedding_similarity(self, query: str, intent_candidates: list[dict]) -> list[dict]:
        """嵌入相似度计算置信度"""
        
        query_embedding = self.encoder.encode(query, normalize_embeddings=True)
        
        scores = []
        for intent in intent_candidates:
            intent_embedding = self.intent_prototypes.get(intent['intent_id'])
            if intent_embedding:
                similarity = cosine_similarity(query_embedding, intent_embedding)
                scores.append({
                    'intent_id': intent['intent_id'],
                    'embedding_confidence': similarity,
                })
        
        return scores
    
    def _fuse_confidence(self, llm_scores: list, embedding_scores: list) -> list[dict]:
        """融合LLM和嵌入置信度"""
        
        # 权重配置
        llm_weight = 0.6
        embedding_weight = 0.4
        
        fused = []
        for llm_item in llm_scores:
            intent_id = llm_item['intent_id']
            llm_conf = llm_item.get('confidence', 0.5)
            
            # 查找对应的嵌入置信度
            emb_conf = 0.5
            for emb_item in embedding_scores:
                if emb_item['intent_id'] == intent_id:
                    emb_conf = emb_item.get('embedding_confidence', 0.5)
                    break
            
            # 加权融合
            fused_conf = llm_weight * llm_conf + embedding_weight * emb_conf
            
            fused.append({
                'intent_id': intent_id,
                'confidence': fused_conf,
                'llm_confidence': llm_conf,
                'embedding_confidence': emb_conf,
                'reason': llm_item.get('reason', ''),
            })
        
        return fused
    
    def _select_primary(self, scores: list[dict]) -> str:
        """选择主意图"""
        if not scores:
            return None
        
        best = max(scores, key=lambda x: x['confidence'])
        return best['intent_id']
```

### 3.3 置信度校准

#### 3.3.1 贝叶斯校准

LLM输出的置信度往往偏高或偏低，需要校准：

```python
# confidence_calibration.py

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

class ConfidenceCalibrator:
    """置信度校准器"""
    
    def __init__(self, method: str = "isotonic"):
        self.method = method
        self.calibrator = None
    
    def fit(self, predictions: list[float], labels: list[int]):
        """训练校准器"""
        # predictions: LLM输出的置信度
        # labels: 实际正确性（1=正确，0=错误）
        
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
        elif self.method == "platt":
            self.calibrator = LogisticRegression()
            # Platt scaling需要logit空间
            predictions = np.log(predictions / (1 - predictions + 1e-8))
        
        self.calibrator.fit(predictions, labels)
    
    def calibrate(self, confidence: float) -> float:
        """校准置信度"""
        if self.calibrator is None:
            return confidence
        
        if self.method == "platt":
            logit = np.log(confidence / (1 - confidence + 1e-8))
            calibrated = self.calibrator.predict_proba([[logit]])[0, 1]
        else:
            calibrated = self.calibrator.predict([confidence])[0]
        
        return np.clip(calibrated, 0, 1)
    
    def evaluate_calibration(self, predictions: list[float], labels: list[int]) -> dict:
        """评估校准质量"""
        from sklearn.metrics import brier_score_loss
        
        # Brier Score：校准误差度量
        brier = brier_score_loss(labels, predictions)
        
        # Expected Calibration Error (ECE)
        ece = self._compute_ece(predictions, labels)
        
        return {
            'brier_score': brier,
            'ece': ece,
            'calibration_quality': 'good' if brier < 0.05 else 'needs_improvement'
        }
    
    def _compute_ece(self, predictions: list[float], labels: list[int], n_bins: int = 10) -> float:
        """计算Expected Calibration Error"""
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0
        
        for i in range(n_bins):
            in_bin = [(p >= bins[i]) and (p < bins[i+1]) for p in predictions]
            if sum(in_bin) > 0:
                avg_conf = np.mean([p for p, in_b in zip(predictions, in_bin) if in_b])
                avg_acc = np.mean([l for l, in_b in zip(labels, in_bin) if in_b])
                ece += abs(avg_conf - avg_acc) * sum(in_bin) / len(predictions)
        
        return ece
```

### 3.4 初始值设置策略

#### 3.4.1 策略矩阵

| 场景 | 初始置信度策略 | 理由 |
|------|---------------|------|
| **高匹配度** | P₀ = 0.85 | 语义高度匹配，但保留修正空间 |
| **中等匹配** | P₀ = 0.65 | 存在歧义，需要更多验证 |
| **低匹配度** | P₀ = 0.35 | 可能是新意图，低置信度触发聚类 |
| **零样本启动** | P₀ = 0.50 | Speech Act 5类，保守估计 |
| **种子数据启动** | P₀ = 原型距离 | 有数据支撑，更精确 |

#### 3.4.2 初始置信度分布

```python
# initial_confidence_distribution.py

class InitialConfidenceStrategy:
    """初始置信度策略"""
    
    def set_initial_confidence(self, 
                               fused_confidence: float,
                               uncertainty_level: str,
                               is_new_intent_candidate: bool = False) -> float:
        """设置初始置信度"""
        
        # 1. 基础置信度（融合值）
        base_conf = fused_confidence
        
        # 2. 不确定性修正
        uncertainty_adjustment = {
            'low': 0.0,
            'medium': -0.1,
            'high': -0.2
        }
        adjustment = uncertainty_adjustment.get(uncertainty_level, 0)
        
        # 3. 新意图候选修正
        if is_new_intent_candidate:
            adjustment -= 0.15  # 新意图降低置信度
        
        # 4. 计算最终置信度
        initial_conf = base_conf + adjustment
        
        # 5. 截断到合理范围
        return np.clip(initial_conf, 0.1, 0.95)  # 避免0和1极端值
```

---

## 四、学习率调度

### 4.1 数学定义

学习率函数：
$$\eta(t) \in \mathbb{R}^+$$ — 第t次更新时的学习率

更新规则：
$$w_{t+1} = w_t + \eta(t) \cdot r(u, c, i) \cdot \nabla L$$

### 4.2 自适应学习率设计

#### 4.2.1 自适应策略

```python
# learning_rate_scheduler.py

from dataclasses import dataclass
from enum import Enum

class LearningRateStrategy(Enum):
    CONSTANT = "constant"
    DECAY = "decay"
    ADAPTIVE = "adaptive"
    STAGE_BASED = "stage_based"

@dataclass
class LearningRateConfig:
    initial_lr: float = 0.1
    min_lr: float = 0.01
    decay_rate: float = 0.95
    warmup_epochs: int = 10
    stability_threshold: float = 0.05

class AdaptiveLearningRateScheduler:
    """自适应学习率调度器"""
    
    def __init__(self, config: LearningRateConfig = None):
        self.config = config or LearningRateConfig()
        self.current_lr = self.config.initial_lr
        self.epoch = 0
        self.performance_history = []
    
    def get_learning_rate(self, 
                          epoch: int,
                          feedback_signal: float,
                          performance_trend: float) -> float:
        """计算自适应学习率"""
        
        # 1. Warmup阶段
        if epoch < self.config.warmup_epochs:
            return self._warmup_lr(epoch)
        
        # 2. 稳定性检测
        if self._is_stable(performance_trend):
            # 稳定状态：降低学习率
            return self.current_lr * self.config.decay_rate
        
        # 3. 信号强度调整
        signal_adjustment = self._signal_based_adjustment(feedback_signal)
        
        # 4. 最终学习率
        self.current_lr = self.current_lr * signal_adjustment
        self.current_lr = max(self.current_lr, self.config.min_lr)
        
        return self.current_lr
    
    def _warmup_lr(self, epoch: int) -> float:
        """Warmup学习率"""
        warmup_ratio = epoch / self.config.warmup_epochs
        return self.config.initial_lr * warmup_ratio
    
    def _is_stable(self, performance_trend: float) -> bool:
        """检测性能稳定性"""
        return abs(performance_trend) < self.config.stability_threshold
    
    def _signal_based_adjustment(self, feedback_signal: float) -> float:
        """基于反馈信号强度调整"""
        # 强信号 → 大调整
        if abs(feedback_signal) > 0.7:
            return 1.2  # 增加20%
        elif abs(feedback_signal) < 0.3:
            return 0.8  # 减少20%
        else:
            return 1.0  # 保持
```

### 4.3 阶段调度策略

#### 4.3.1 四阶段调度

```
┌─────────────────────────────────────────────────────────────┐
│               学习率四阶段调度策略                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Phase 1: Warmup (0-10 epochs)                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  η(t) = η₀ × t/T_warmup                             │   │
│  │  目的：避免初期不稳定                                │   │
│  │  推荐：η₀ = 0.1 → 逐步提升到 0.1                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Phase 2: Exploration (10-50 epochs)                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  η(t) = η₀                                         │   │
│  │  目的：探索最优参数空间                              │   │
│  │  推荐：η₀ = 0.1                                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Phase 3: Refinement (50-100 epochs)                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  η(t) = η₀ × decay_rate^(t-T_explore)              │   │
│  │  目的：精细调整，减少震荡                            │   │
│  │  推荐：decay_rate = 0.95                            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Phase 4: Convergence (100+ epochs)                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  η(t) = η_min                                      │   │
│  │  目的：稳定收敛                                      │   │
│  │  推荐：η_min = 0.01                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 4.3.2 阶段调度实现

```python
# stage_scheduler.py

class StageLearningRateScheduler:
    """阶段学习率调度器"""
    
    def __init__(self):
        self.stages = {
            'warmup': {'epochs': (0, 10), 'lr_func': self._warmup_lr},
            'exploration': {'epochs': (10, 50), 'lr_func': self._constant_lr},
            'refinement': {'epochs': (50, 100), 'lr_func': self._decay_lr},
            'convergence': {'epochs': (100, float('inf')), 'lr_func': self._min_lr}
        }
        
        self.lr_params = {
            'initial_lr': 0.1,
            'min_lr': 0.01,
            'decay_rate': 0.95,
        }
    
    def get_learning_rate(self, epoch: int) -> float:
        """根据阶段获取学习率"""
        
        # 确定当前阶段
        current_stage = self._determine_stage(epoch)
        stage_config = self.stages[current_stage]
        
        # 调用阶段特定的学习率函数
        return stage_config['lr_func'](epoch, stage_config['epochs'])
    
    def _determine_stage(self, epoch: int) -> str:
        """确定当前阶段"""
        for stage_name, config in self.stages.items():
            if config['epochs'][0] <= epoch < config['epochs'][1]:
                return stage_name
        return 'convergence'
    
    def _warmup_lr(self, epoch: int, epochs_range: tuple) -> float:
        """Warmup阶段学习率"""
        warmup_epochs = epochs_range[1] - epochs_range[0]
        ratio = epoch / warmup_epochs
        return self.lr_params['initial_lr'] * ratio
    
    def _constant_lr(self, epoch: int, epochs_range: tuple) -> float:
        """探索阶段固定学习率"""
        return self.lr_params['initial_lr']
    
    def _decay_lr(self, epoch: int, epochs_range: tuple) -> float:
        """精调阶段衰减学习率"""
        refinement_start = epochs_range[0]
        decay_epochs = epoch - refinement_start
        return self.lr_params['initial_lr'] * (self.lr_params['decay_rate'] ** decay_epochs)
    
    def _min_lr(self, epoch: int, epochs_range: tuple) -> float:
        """收敛阶段最小学习率"""
        return self.lr_params['min_lr']
```

### 4.4 反馈信号强度学习率

#### 4.4.1 动态调整规则

| 反馈信号强度 | 学习率调整 | 说明 |
|-------------|----------|------|
| |r| > 0.7 | η × 1.2 | 强信号，快速响应 |
| |r| ∈ [0.5, 0.7] | η × 1.0 | 正常信号 |
| |r| ∈ [0.3, 0.5] | η × 0.9 | 弱信号，谨慎调整 |
| |r| < 0.3 | η × 0.8 | 微弱信号，保持稳定 |

#### 4.4.2 实现

```python
class SignalBasedLearningRateAdjuster:
    """基于反馈信号强度的学习率调整"""
    
    def adjust_by_signal(self, base_lr: float, feedback_signal: float) -> float:
        """根据信号强度调整学习率"""
        
        signal_magnitude = abs(feedback_signal)
        
        if signal_magnitude > 0.7:
            multiplier = 1.2
        elif signal_magnitude > 0.5:
            multiplier = 1.0
        elif signal_magnitude > 0.3:
            multiplier = 0.9
        else:
            multiplier = 0.8
        
        return base_lr * multiplier
```

---

## 五、参数获取完整流程

### 5.1 系统初始化流程

```python
# system_parameter_initialization.py

class SystemParameterInitializer:
    """系统参数初始化器"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # 初始化各模块
        self.chunk_encoder = ChunkProcessor(config['chunk_encoder'])
        self.intent_encoder = IntentEncoder(config['intent_encoder'])
        self.feedback_fusion = FeedbackFusionEngine(config['feedback_weights'])
        self.confidence_initializer = ConfidenceInitializer(
            config['llm_client'],
            config['intent_encoder']
        )
        self.confidence_calibrator = ConfidenceCalibrator(config['calibration_method'])
        self.lr_scheduler = AdaptiveLearningRateScheduler(config['lr_config'])
    
    def initialize(self, seed_data: dict = None) -> dict:
        """完整初始化流程"""
        
        # Step 1: 初始化嵌入编码器
        chunk_config = self._init_chunk_encoder()
        intent_prototypes = self.intent_encoder.initialize_prototypes(seed_data)
        
        # Step 2: 初始化置信度系统
        if seed_data:
            # 使用种子数据训练校准器
            self._train_calibrator(seed_data)
        
        # Step 3: 初始化学习率
        initial_lr = self.lr_scheduler.get_learning_rate(0, 0, 0)
        
        # Step 4: 验证初始化
        validation_result = self._validate_initialization()
        
        return {
            'chunk_encoder_config': chunk_config,
            'intent_prototypes': intent_prototypes,
            'confidence_calibrator': self.confidence_calibrator,
            'initial_learning_rate': initial_lr,
            'validation_result': validation_result,
            'initialization_time': datetime.now(),
        }
    
    def _init_chunk_encoder(self) -> dict:
        """初始化Chunk编码器"""
        return {
            'model': self.chunk_encoder.encoder_name,
            'dimension': self.chunk_encoder.encoder.get_sentence_embedding_dimension(),
            'normalize': True,
            'batch_size': 32,
        }
    
    def _train_calibrator(self, seed_data: dict):
        """训练置信度校准器"""
        # 从种子数据提取预测和标签
        predictions = []
        labels = []
        
        for intent_id, samples in seed_data.items():
            for sample in samples:
                pred = self.intent_encoder.encode_intent(sample['text'])
                # 计算与原型距离作为预测置信度
                prototype = self.intent_prototypes[intent_id]
                conf = cosine_similarity(pred, prototype)
                predictions.append(conf)
                labels.append(1 if sample['verified'] else 0)
        
        self.confidence_calibrator.fit(predictions, labels)
    
    def _validate_initialization(self) -> dict:
        """验证初始化结果"""
        test_queries = [
            "公司的请假流程是什么？",
            "请帮我查询报销状态",
            "谢谢你的帮助！"
        ]
        
        results = []
        for query in test_queries:
            # 编码
            embedding = self.intent_encoder.encode_intent(query)
            
            # 计算与原型距离
            distances = {}
            for intent_id, prototype in self.intent_encoder.intent_prototypes.items():
                distances[intent_id] = cosine_similarity(embedding, prototype)
            
            results.append({
                'query': query,
                'embedding_dim': len(embedding),
                'best_intent': max(distances, key=distances.get),
                'best_distance': max(distances.values()),
            })
        
        return {
            'test_results': results,
            'status': 'success' if all(r['best_distance'] > 0.3 for r in results) else 'needs_adjustment'
        }
```

### 5.2 运行时参数获取

```python
# runtime_parameter_acquisition.py

class RuntimeParameterAcquirer:
    """运行时参数获取器"""
    
    def __init__(self, system_initializer: SystemParameterInitializer):
        self.system = system_initializer
    
    def process_query(self, 
                      query: str,
                      context: dict = None) -> ProcessingResult:
        """处理查询并获取参数"""
        
        # 1. 获取Chunk嵌入 φ(c)
        chunk_embeddings = self.system.chunk_encoder.process_content(query)
        
        # 2. 获取Intent嵌入 ψ(i)
        intent_embedding = self.system.intent_encoder.encode_intent(query)
        
        # 3. 初始化置信度 P(i|c)
        intent_candidates = self._get_intent_candidates()
        confidence_result = self.system.confidence_initializer.initialize_confidence(
            query, intent_candidates
        )
        
        # 4. 校准置信度
        calibrated_confidence = self.system.confidence_calibrator.calibrate(
            confidence_result['intent_scores'][0]['confidence']
        )
        
        # 5. 获取当前学习率
        current_lr = self.system.lr_scheduler.get_learning_rate(
            self.system.epoch, 0, 0
        )
        
        return ProcessingResult(
            chunk_embedding=chunk_embeddings[0].embedding if chunk_embeddings else None,
            intent_embedding=intent_embedding,
            confidence=calibrated_confidence,
            learning_rate=current_lr,
            primary_intent=confidence_result['primary_intent'],
        )
    
    def process_feedback(self,
                         query: str,
                         response: str,
                         user_feedback: dict) -> float:
        """处理反馈并获取信号"""
        
        # 解析各类反馈
        explicit_feedback = self._parse_explicit(user_feedback)
        implicit_signals = self._detect_implicit(query, response, user_feedback)
        cot_signals = self._extract_cot(user_feedback)
        
        # 融合反馈信号
        r_total = self.system.feedback_fusion.fuse(
            explicit_feedback, implicit_signals, cot_signals
        )
        
        return r_total
    
    def update_parameters(self,
                         feedback_signal: float,
                         confidence: float) -> dict:
        """基于反馈更新参数"""
        
        # 获取自适应学习率
        current_epoch = self.system.epoch
        performance_trend = self._compute_performance_trend()
        
        adaptive_lr = self.system.lr_scheduler.get_learning_rate(
            current_epoch, feedback_signal, performance_trend
        )
        
        # 更新置信度（假设存在参数w）
        # w_new = w_old + η × r × ∇L
        # 这里简化为置信度调整
        confidence_update = adaptive_lr * feedback_signal
        new_confidence = confidence + confidence_update
        new_confidence = np.clip(new_confidence, 0.1, 0.95)
        
        return {
            'learning_rate': adaptive_lr,
            'confidence_update': confidence_update,
            'new_confidence': new_confidence,
            'epoch': current_epoch + 1,
        }
```

---

## 六、参数配置清单

### 6.1 完整配置文件

```yaml
# system_parameters_config.yaml

# 内容-意图映射系统参数配置

embedding:
  chunk_encoder:
    model: "bge-m3"
    dimension: 1024
    normalize: true
    batch_size: 32
    max_length: 512
    overlap: 50
    strategy: "semantic"
    quantization: "int8"
    cache_enabled: true
    
  intent_encoder:
    model: "bge-m3"
    dimension: 1024
    normalize: true
    initialization: "zero_shot"  # 或 "seed_data"
    
feedback:
  weights:
    explicit: 0.5
    implicit: 0.3
    cot: 0.2
  
  explicit_mapping:
    thumbs_up: 1.0
    thumbs_down: -1.0
    star_5: 1.0
    star_4: 0.8
    star_3: 0.5
    star_2: -0.5
    star_1: -1.0
  
  implicit_detection:
    follow_up_threshold: 0.3
    topic_shift_threshold: 0.5
    reading_time_ratio_low: 0.3
    reading_time_ratio_high: 1.5
  
  cot_intensity:
    correction_minor: -0.5
    correction_major: -0.8
    correction_fundamental: -1.0
    supplement_small: 0.2
    supplement_medium: 0.3
    supplement_large: 0.5
    interruption: -0.3
    confirmation: 0.2

confidence:
  initialization:
    llm_weight: 0.6
    embedding_weight: 0.4
    
  calibration:
    method: "isotonic"  # 或 "platt"
    target_brier: 0.05
    
  initial_values:
    high_match: 0.85
    medium_match: 0.65
    low_match: 0.35
    zero_shot: 0.50
    
learning_rate:
  initial_lr: 0.1
  min_lr: 0.01
  decay_rate: 0.95
  
  stages:
    warmup:
      epochs: 0-10
      strategy: "linear_increase"
    exploration:
      epochs: 10-50
      strategy: "constant"
    refinement:
      epochs: 50-100
      strategy: "exponential_decay"
    convergence:
      epochs: 100+
      strategy: "min_constant"
  
  signal_adjustment:
    strong_signal: 1.2   # |r| > 0.7
    normal_signal: 1.0   # |r| ∈ [0.5, 0.7]
    weak_signal: 0.9     # |r| ∈ [0.3, 0.5]
    micro_signal: 0.8    # |r| < 0.3

validation:
  target_metrics:
    semantic_fidelity: 0.85
    signal_effectiveness: 0.70
    calibration_error: 0.05
    convergence_epochs: 100
```

---

## 七、验证指标

### 7.1 指标清单

| 维度 | 指标 | 目标值 | 测量方法 |
|------|------|-------|---------|
| **嵌入质量** | 语义保真度 | > 0.85 | 压缩前后相似度 |
| **嵌入质量** | 聚类纯度 | > 0.80 | HDBSCAN聚类评估 |
| **反馈信号** | 信号有效比 | > 70% | 反馈→性能改善率 |
| **反馈信号** | 信号检测准确率 | > 85% | 隐式行为检测验证 |
| **置信度** | Brier Score | < 0.05 | 校准误差 |
| **置信度** | ECE | < 0.1 | Expected Calibration Error |
| **学习率** | 收敛速度 | < 100 epochs | 参数稳定化 |
| **学习率** | 最终准确率 | > 90% | 意图分类准确率 |

### 7.2 验证代码

```python
# parameter_validation.py

class ParameterValidator:
    """参数验证器"""
    
    def validate_all(self, system: SystemParameterInitializer) -> ValidationReport:
        """完整参数验证"""
        
        results = {
            'embedding_quality': self._validate_embedding(system),
            'feedback_signal': self._validate_feedback(system),
            'confidence': self._validate_confidence(system),
            'learning_rate': self._validate_learning_rate(system),
        }
        
        return ValidationReport(
            results=results,
            overall_status=self._determine_status(results),
            recommendations=self._generate_recommendations(results)
        )
    
    def _validate_embedding(self, system) -> dict:
        """验证嵌入质量"""
        # 测试样本
        test_samples = [
            ("请假流程", "T_RULES"),
            ("报销查询", "T_RESOURCES"),
            ("谢谢帮助", "L_EXPRESSIVE"),
        ]
        
        # 计算语义保真度
        fidelity_scores = []
        for query, expected_intent in test_samples:
            embedding = system.intent_encoder.encode_intent(query)
            prototype = system.intent_encoder.intent_prototypes.get(expected_intent)
            if prototype:
                fidelity = cosine_similarity(embedding, prototype)
                fidelity_scores.append(fidelity)
        
        avg_fidelity = np.mean(fidelity_scores)
        
        return {
            'semantic_fidelity': avg_fidelity,
            'target': 0.85,
            'status': 'pass' if avg_fidelity > 0.85 else 'fail'
        }
    
    def _validate_feedback(self, system) -> dict:
        """验证反馈信号"""
        # 模拟反馈测试
        test_feedback = [
            {'explicit': [{'type': 'thumbs_up'}], 'expected': 0.5},
            {'implicit': [{'type': 'follow_up', 'detected': True}], 'expected': 0.075},
        ]
        
        signal_effectiveness = []
        for feedback in test_feedback:
            # 计算实际信号
            r = system.feedback_fusion.fuse(
                feedback.get('explicit', []),
                feedback.get('implicit', []),
                []
            )
            expected = feedback['expected']
            effectiveness = abs(r - expected) < 0.1
            signal_effectiveness.append(effectiveness)
        
        return {
            'signal_effectiveness': np.mean(signal_effectiveness),
            'target': 0.70,
            'status': 'pass' if np.mean(signal_effectiveness) > 0.70 else 'fail'
        }
    
    def _validate_confidence(self, system) -> dict:
        """验证置信度系统"""
        if system.confidence_calibrator.calibrator is None:
            return {'status': 'not_calibrated', 'recommendation': '需要训练校准器'}
        
        # 使用历史数据评估
        predictions = [...]  # 历史预测
        labels = [...]       # 历史标签
        
        metrics = system.confidence_calibrator.evaluate_calibration(predictions, labels)
        
        return {
            'brier_score': metrics['brier_score'],
            'ece': metrics['ece'],
            'target_brier': 0.05,
            'target_ece': 0.1,
            'status': metrics['calibration_quality']
        }
    
    def _validate_learning_rate(self, system) -> dict:
        """验证学习率调度"""
        # 模拟训练过程
        lr_history = []
        for epoch in range(150):
            lr = system.lr_scheduler.get_learning_rate(epoch, 0, 0)
            lr_history.append(lr)
        
        # 检查收敛
        final_lr_stability = np.std(lr_history[-20:])
        
        return {
            'lr_decay_trajectory': lr_history,
            'final_lr': lr_history[-1],
            'stability': final_lr_stability,
            'target_min_lr': 0.01,
            'status': 'pass' if final_lr_stability < 0.005 else 'needs_tuning'
        }
```

---

## 八、总结

### 8.1 参数获取方法总结

| 参数 | 获取方法 | 核心工具 | 验证指标 |
|------|---------|---------|---------|
| φ(c) Chunk嵌入 | 预训练编码器 + 语义分块 | bge-m3 | 语义保真度 > 0.85 |
| ψ(i) Intent嵌入 | 零样本原型 / 种子数据原型 | bge-m3 + 原型中心 | 聚类纯度 > 0.80 |
| r_explicit 显式反馈 | 数值映射表 | 固定规则 | — |
| r_implicit 隐式反馈 | 语义距离检测 + 行为分析 | encoder + 行为检测 | 检测准确率 > 85% |
| r_cot CoT信号 | LLM分析 + 强度估计 | LLM + 分级映射 | 信号有效比 > 70% |
| P(i|c) 置信度 | LLM推断 + 嵌入相似度 + 校准 | LLM + calibrator | Brier < 0.05 |
| η(t) 学习率 | 四阶段调度 + 信号自适应 | scheduler | 收敛 < 100 epochs |

### 8.2 推荐配置

**企业应用推荐配置**：
- Chunk编码器：bge-m3 (1024维，INT8量化)
- Intent编码器：bge-m3 (零样本启动 → 种子数据精细化)
- 反馈权重：explicit=0.5, implicit=0.3, cot=0.2
- 置信度校准：Isotonic Regression
- 学习率：四阶段调度，初始0.1，最小0.01

### 8.3 实施路径

```
Phase 1 (Day 0-7): 嵌入系统初始化
  ├─ 加载预训练编码器
  ├─ 初始化Intent原型（零样本）
  └─ 验证嵌入质量

Phase 2 (Day 7-14): 反馈系统部署
  ├─ 显式反馈接入
  ├─ 隐式反馈检测部署
  ├─ CoT信号提取集成
  └─ 反馈融合验证

Phase 3 (Day 14-21): 置信度系统训练
  ├─ 收集种子数据
  ├─ 训练校准器
  └─ 验证校准质量

Phase 4 (Day 21-30): 学习率调优
  ├─ 四阶段调度部署
  ├─ 信号自适应调整集成
  └─ 收敛性验证
```

---

## 附录：参数获取流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                系统参数获取完整流程                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │   用户查询   │ ──→ │   内容分块   │ ──→ │  Chunk编码  │       │
│  │   Query     │     │  Chunking   │     │    φ(c)     │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│                                                 │               │
│                                                 ▼               │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │  Intent     │ ──→ │  原型匹配   │ ──→ │ Intent编码  │       │
│  │  Schema     │     │ Prototype   │     │    ψ(i)     │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│                                                 │               │
│                                                 ▼               │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │   LLM推断   │ ──→ │  嵌入相似度  │ ──→ │  置信度融合  │       │
│  │  Inference  │     │ Similarity  │     │   P(i|c)    │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│                                                 │               │
│                                                 ▼               │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │  用户响应   │ ──→ │  反馈检测   │ ──→ │  反馈融合   │       │
│  │  Response   │     │ Detection   │     │  r(u,c,i)   │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│                                                 │               │
│                                                 ▼               │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │  阶段判断   │ ──→ │  信号调整   │ ──→ │  学习率计算  │       │
│  │   Stage     │     │  Signal Adj │     │    η(t)     │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│                                                 │               │
│                                                 ▼               │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              参数更新：w_new = w_old + η × r × ∇L     │       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

**文档生成时间**: 2026-04-03 00:53 GMT+8  
**团队**: content-intent-math  
**作者**: agent-system