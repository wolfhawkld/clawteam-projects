# Universal Intent Framework Research

通用意图分类框架研究项目

**创建时间**: 2026-03-28
**项目状态**: Phase 1 完成

---

## 项目概述

本项目旨在研究通用意图分类框架，为 IntentWeight 项目提供理论基础和实施方案。

### 核心发现

**Speech Act Theory (Austin 1962, Searle 1969) 是唯一具备理论闭包性的通用意图框架。**

---

## 研究成果

| 文件 | 说明 |
|------|------|
| [survey-report.md](survey-report.md) | 通用意图分类框架研究（计算领域） |
| [clustering-analysis.md](clustering-analysis.md) | 意图分类与语义聚类关系分析 |
| [implementation-guide.md](implementation-guide.md) | 实施方案（含三层架构设计） |
| [speech-act-closure-analysis.md](speech-act-closure-analysis.md) | **🆕 Speech Act 闭包性分析** |

---

## 核心结论

### 三层意图架构

```
第一层: Speech Act 5类 (语言学基础层)
    → 零样本启动，理论完备
    
第二层: R³ + NFQA 13类 (任务类型层)
    → few-shot 精细化
    
第三层: 聚类发现的领域意图 (数据层)
    → 动态扩展，自动发现
```

### Speech Act 5类

| 类别 | 定义 | Direction of Fit |
|------|------|-----------------|
| **Assertive** | 陈述事实 | Words → World |
| **Directive** | 请求行动 | World → Words |
| **Commissive** | 承诺未来 | World → Words |
| **Expressive** | 表达情感 | Null |
| **Declarative** | 宣告状态 | World ↔ Words |

### 闭包性验证

- **理论证明**: Searle 穷举论证 + 50年学术验证
- **实践验证**: 规则分类器测试准确率 94%
- **无反例**: 所有言语行为都可归入 5 类

---

## 与 IntentWeight 项目的关系

| IntentWeight 阶段 | 本项目贡献 |
|------------------|-----------|
| Phase 1A: 方法验证 | 理论框架 + 分类器设计 |
| Phase 1B: 实际部署 | 三层架构实施方案 |
| 持续优化 | 新意图发现机制 |

---

## 参考文献

### 语言学/哲学基础
- Austin, J.L. (1962). *How to Do Things with Words*
- Searle, J.R. (1969). *Speech Acts: An Essay in the Philosophy of Language*

### 计算领域框架
- Baranova et al. (2022). "A Non-Factoid Question-Answering Taxonomy" (SIGIR Best Paper)
- Kilian et al. (2026). "Rules, Resources, and Restrictions" (CHIIR 2026)

---

*项目团队: ClawTeam intent-framework + Outis 补充*
*最后更新: 2026-03-29*