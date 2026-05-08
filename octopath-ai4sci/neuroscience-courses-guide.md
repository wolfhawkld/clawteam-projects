# 计算神经科学与认知科学在线课程指南

> 三门精心挑选的 Coursera/edX 课程，从单神经元建模到认知动力学，建立计算神经科学的完整知识体系。
> 无需实验室，全程在线，适合 CS/AI 背景的在职工程师自学。

---

## 推荐学习路线

```
Phase 1: Neuronal Dynamics (EPFL)  → 单神经元基础，7周
Phase 2: Computational Neuroscience (UW) → 神经网络计算原理，8周
Phase 3: Neuronal Dynamics of Cognition (EPFL) → 认知的神经动力学，7周
```

总计约 24 周（6 个月），每周 4-6 小时。

---

## 课程一：Neuronal Dynamics

| 项目 | 详情 |
|------|------|
| **平台** | edX |
| **学校** | EPFL（瑞士洛桑联邦理工学院） |
| **教授** | Wulfram Gerstner（计算神经科学权威） |
| **时长** | 7 周，每周 4-6 小时 |
| **难度** | 入门-中级 |
| **费用** | 免费旁听 / $169 证书 |
| **链接** | https://www.edx.org/learn/neuroscience/epfl-neuronal-dynamics |

### 内容概要

- Hodgkin-Huxley 神经元模型
- 突触传递与可塑性
- STDP（脉冲时间依赖可塑性）——Hebbian 学习的生物学基础
- 神经元群体编码（population coding）
- 强化学习在神经元层面的应用

### 为什么从这门开始

这是计算神经科学的"Hello World"。学完你会理解：一个真实的生物神经元如何用数学建模，可塑性如何实现"学习"，以及为什么深度学习中的 Hebbian 学习、奖励信号机制在生物学中有精确的对应物。

---

## 课程二：Computational Neuroscience

| 项目 | 详情 |
|------|------|
| **平台** | Coursera |
| **学校** | University of Washington |
| **时长** | 8 周，每周 4-6 小时 |
| **难度** | 中级 |
| **评分** | ⭐4.6（1,100+ 评价） |
| **费用** | 免费旁听 / $49/月证书 |
| **链接** | https://www.coursera.org/learn/computational-neuroscience |

### 内容概要

- 视觉系统的计算原理（视网膜 → V1 → 高级视觉皮层）
- 感觉-运动控制回路
- 学习与记忆的神经网络模型
- 信息论在神经编码中的应用
- 编程作业：用 MATLAB/Python 写神经元模拟代码

### 为什么这门重要

这门课把"计算"和"神经系统"直接连接起来。你学到的视觉系统层级处理模型（retina → V1 → V2 → IT）就是 CNN 的生物对应物。理解这些会让你对 Transformer 为什么 work、为什么需要层级抽象有更深的认识。

---

## 课程三：Neuronal Dynamics of Cognition

| 项目 | 详情 |
|------|------|
| **平台** | edX |
| **学校** | EPFL |
| **教授** | Wulfram Gerstner |
| **时长** | 7 周，每周 4-6 小时 |
| **难度** | 中高级（需要微积分和微分方程） |
| **费用** | 免费旁听 / $169 证书 |
| **链接** | https://www.edx.org/learn/neuroscience/epfl-computational-neuroscience-neuronal-dynamics-of-cognition |

### 内容概要

- 数千神经元的集体动力学
- 平均场理论（mean-field theory）
- 非线性微分方程在神经建模中的应用
- 记忆形成模型（attractor networks）
- 皮层场模型与感知
- 决策的神经计算基础

### 为什么最后一门

这是整个路线的顶峰：从单个神经元到神经元群体，再到"认知"的涌现。学到这里你会接触一个核心思想——**预测编码理论**（predictive coding），即大脑本质上是一个预测机器，不断预测感官输入并修正误差。这正是 BERT 的 masked prediction 的生物学灵感来源，也是理解当前 AI 发展最深层的理论框架。

---

## 对你 AI 工作的实际价值

| 课程 | 对应的 AI 概念 |
|------|--------------|
| Neuronal Dynamics | Hebbian 学习 → 无监督预训练；STDP → 时序建模 |
| Computational Neuroscience | 层级处理 → CNN/Transformer 层级设计；感受野 → attention 机制 |
| Neuronal Dynamics of Cognition | 预测编码 → BERT MLM；attractor networks → Hopfield 网络 / 现代 RNN |

---

## 更多认知科学课程（备选）

### Coursera: Philosophy and the Sciences — Introduction to the Philosophy of Cognitive Sciences

- **学校**：University of Edinburgh
- **内容**：从哲学角度审视认知科学——意识、表征、具身认知
- **费用**：免费旁听 / $49/月证书
- **适合**：对"AI 是否有意识？""什么是理解？"等基础问题感兴趣时选

### edX: Cognitive Science (多门课程)

- **学校**：UC San Diego, MIT 等
- **内容**：认知科学入门、语言与认知、决策科学
- **费用**：免费旁听

### Coursera: Computational Neuroscience 专题

- **学校**：多个学校
- **内容**：包含上面 UW 的课 + 神经影像学 + 机器学习的神经科学应用
- **费用**：$49/月

---

## 学习建议

1. **不要同时报名**：先旁听免费版，觉得有价值再付费拿证书
2. **配编程时间**：Neuronal Dynamics 和 UW 的课有编程作业，用 Python 就行
3. **结合 Claude/GPT 辅助**：遇到数学推导看不懂，直接让 AI 解释

---

## 学位 vs 课程的取舍

如果目标是**理解认知科学并用它指导 AI 工作**（比如你的 IntentWeight 论文中涉及的流形假设），这三门课足够了。如果需要学位背书，目前最可行的是：

- **Harvard Extension Psychology**（混合模式，$35k，有认知科学方向）
- 或者走**科研路线**直接用 CS 背景发计算神经科学/AI 交叉论文

---

*创建于 2026-05-04 · UT Austin MSAI 分析项目 · Nemesis*
