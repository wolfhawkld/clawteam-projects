# 内容-意图映射系统：数学形式化文档

**Created**: 2026-04-03  
**Type**: Mathematical Formalization  
**Status**: Complete

---

## 概述

本文档集合定义了内容-意图映射系统的完整数学形式化，包括：

1. **参数配置指南** — 核心公式参数化、超参数推荐值、敏感性分析
2. **系统参数获取指南** — 嵌入向量、反馈信号、置信度、学习率的获取方式
3. **端到端流程图** — Mermaid流程图 + 节点公式对照表

---

## 文档索引

| 文档 | 内容 | 页数 |
|------|------|------|
| [01-parameter-configuration-guide.md](01-parameter-configuration-guide.md) | 参数配置与调参指南 | ~30页 |
| [02-system-parameter-acquisition-guide.md](02-system-parameter-acquisition-guide.md) | 系统参数获取实现指南 | ~25页 |
| [03-content-intent-flowchart.md](03-content-intent-flowchart.md) | 端到端流程图 | ~10页 |

---

## 核心数学框架速览

### 1. 映射函数

$$f_\theta(c, i) = \sigma(\phi(c)^T W \psi(i))$$

- $\phi(c)$: Chunk编码器，输出维度 $d_c$
- $\psi(i)$: Intent编码器，输出维度 $d_i$
- $W$: 映射权重矩阵
- $\sigma$: Sigmoid激活

### 2. 贝叶斯更新

$$p_{t+1}(i|c) \propto p_t(i|c) \times \mathcal{L}(\text{feedback} | i, c)$$

- $p_t(i|c)$: $t$ 时刻意图先验
- $\mathcal{L}$: 似然函数（反馈信号的概率模型）

### 3. RL更新

$$\text{conf}(c, i) \leftarrow \text{conf}(c, i) + \alpha [R - \text{conf}(c, i)]$$

- $\alpha$: 学习率
- $R$: 奖励信号

### 4. 多信号融合

$$R_{total} = \sum_k w_k \times R_k$$

- $w_k$: 第 $k$ 类信号权重
- $R_k$: 第 $k$ 类信号奖励值

### 5. 学习率调度

$$\eta(t) = \begin{cases}
\eta_0 \cdot t/T_{warmup} & t < T_{warmup} \\
\eta_0 & T_{warmup} \le t < T_{explore} \\
\eta_0 \cdot \gamma^{t-T_{explore}} & T_{explore} \le t < T_{refine} \\
\eta_{min} & t \ge T_{refine}
\end{cases}$$

---

## 参数速查表

| 参数 | 符号 | 冷启动 | 稳定期 | 高噪声 | 非平稳 |
|------|------|--------|--------|--------|--------|
| 学习率 | $\alpha$ | 0.4 | 0.08 | 0.02 | 0.20 |
| 确认似然 | $\alpha_{似然}$ | 0.95 | 0.90 | 0.80 | 0.88 |
| 拒绝似然 | $\beta_{似然}$ | 0.05 | 0.10 | 0.15 | 0.12 |
| 温度 | $T$ | 0.5 | 1.0 | 1.5 | 0.8 |
| 探索率 | $\epsilon$ | 0.3 | 0.05 | 0.02 | 0.15 |

---

## 流程图核心节点

```
初始化 → 检索 → 反馈收集 → 映射更新 → 监控评估
   │        │        │          │          │
   ▼        ▼        ▼          ▼          ▼
P(i|c)   σ(W·[e_q;e_c])  Σw_k·s_k  θ←θ+η∇J  KL(P_t||P_t-1)
```

---

## 使用建议

1. **冷启动**：使用 `parameter-configuration-guide.md` 中的"冷启动场景"配置
2. **实施**：按照 `system-parameter-acquisition-guide.md` 的实现指南开发
3. **调试**：参考 `content-intent-flowchart.md` 的流程图定位问题节点
4. **调参**：使用参数敏感性分析确定优先调优的参数

---

*本文档由 ClawTeam content-intent-math 团队协作生成*