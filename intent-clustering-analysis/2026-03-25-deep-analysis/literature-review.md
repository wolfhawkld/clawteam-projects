# Literature Review: Intent Clustering and Knowledge Graph RAG Systems

**Author:** literature-surveyor  
**Date:** 2026-03-25  
**Team:** intent-clustering-team

---

## Executive Summary

This literature review analyzes existing work in four key research areas relevant to intent clustering in knowledge graph RAG systems: (1) Intent-Aware Recommendation, (2) Knowledge Graph Neural Networks, (3) Multi-Interest Modeling, and (4) Retrieval-Augmented Generation. The review identifies research gaps and opportunities for the proposed approach combining ReAct Agent, user feedback, unsupervised intent clustering, and dynamic weighting.

---

## 1. Intent-Aware Recommendation Systems

### 1.1 IKGR: LLM-based Intent Knowledge Graph Recommender

**Citation:** Zheng, W. et al. (2025). "Tuning-Free LLM Can Build A Strong Recommender Under Sparse Connectivity And Knowledge Gap Via Extracting Intent." arXiv:2505.10900. Accepted at Learning on Graphs (LoG) 2025.

**Key Contributions:**
- Constructs an **intent-centric knowledge graph** where both users and items are explicitly linked to intent nodes extracted by a tuning-free, RAG-guided LLM pipeline
- Introduces **mutual-intent connectivity densification strategy** to shorten semantic paths between users and long-tail items
- Employs lightweight GNN layer on top of intent-enhanced graph for low-latency recommendations

**Strengths:**
- Addresses sparsity and cold-start problems through intent-based connectivity
- Grounds intents in external knowledge sources and user profiles
- Fully offline LLM pipeline ensures efficiency

**Limitations:**
- Intent extraction relies on LLM capabilities without explicit user feedback loop
- Static intent representation once extracted
- No dynamic weight adjustment based on user interactions

### 1.2 IKG-EMR: Intent-Aware Knowledge Graph-based Model

**Citation:** Intent-aware knowledge graph-based model for electrical power material recommendation. PeerJ Computer Science, 2025. PMC12453814.

**Key Contributions:**
- Constructs **User-Item-Topic tripartite graph** for modeling user intent
- Combines GNN for intent embedding with Transformer for preference extraction
- **Adaptive fusion with attention network** to integrate user preference and intent features

**Architecture:**
```
User-Item-Topic Tripartite Graph → GNN → Intent Embedding
User Behavior Sequence → Transformer → Preference Embedding
Adaptive Fusion Network → Final User Representation
```

**Strengths:**
- Explicit modeling of user intent as coarse-grained embedding vs. fine-grained preference
- Tripartite graph structure alleviates sparsity issues
- Demonstrates significant improvement over KGAT, CKAN, KGCN baselines

**Limitations:**
- Intent derived from topic modeling (LDA) rather than explicit user intent
- No real-time feedback mechanism
- Domain-specific (electrical power materials), generalizability unclear

---

## 2. Knowledge Graph Neural Networks for Recommendation

### 2.1 KGAT: Knowledge Graph Attention Network

**Citation:** Wang, X. et al. (2019). "KGAT: Knowledge Graph Attention Network for Recommendation." KDD 2019.

**Key Concepts:**
- Propagates user preferences via knowledge graph relationships
- Uses attention mechanism to distinguish importance of different neighbors
- High-order connectivity captures multi-hop user-item relations

**Relevance to Intent Clustering:**
- Provides foundation for graph-based preference propagation
- Attention mechanism could be adapted for intent-aware weighting
- However, lacks explicit intent modeling

### 2.2 Related GNN-based Methods

| Method | Key Feature | Intent Modeling |
|--------|-------------|-----------------|
| **GCMC** | Graph Convolutional Matrix Completion | ❌ No explicit intent |
| **SR-GNN** | Session-based GNN | ❌ Implicit only |
| **KGCN** | KG + CNN for recommendation | ❌ No intent nodes |
| **CKAN** | Collaborative KG Attention Network | ❌ No intent hierarchy |

**Research Gap:** Existing GNN-based methods aggregate information from neighboring nodes but fail to explicitly capture and reason about user intent as a first-class entity in the graph.

---

## 3. Multi-Interest Modeling

### 3.1 MIND: Multi-Interest Network with Dynamic Routing

**Citation:** Li, C. et al. (2019). "Multi-Interest Network with Dynamic Routing for Recommendation at Tmall." CIKM 2019. arXiv:1904.08030.

**Key Contributions:**
- Represents users with **multiple vectors** encoding different interest aspects
- **Capsule routing mechanism** for clustering historical behaviors
- **Label-aware attention** for learning multi-vector user representation

**Relevance:**
- Multi-interest modeling aligns with multi-intent scenarios
- Dynamic routing mechanism adaptable for intent clustering
- Deployed at Tmall homepage, demonstrating scalability

**Limitations:**
- Interests learned from behavior sequences, not explicit intents
- No knowledge graph integration
- Static routing weights, no dynamic adjustment

### 3.2 Other Multi-Interest Approaches

- **MCPRN**: Mixture-channel purpose routing networks for session-based recommendation
- **MINT**: Multi-intent translation graph neural network
- **DSSRec**: Disentangled sequential recommendation with intent variables

**Research Gap:** Multi-interest methods cluster behaviors into interest groups but don't connect to knowledge graph entities or support real-time intent refinement through user feedback.

---

## 4. Retrieval-Augmented Generation (RAG)

### 4.1 RAG Survey

**Citation:** Gao, Y. et al. (2024). "Retrieval-Augmented Generation for Large Language Models: A Survey." arXiv:2312.10997.

**Key Paradigms:**
1. **Naive RAG**: Simple retrieve-then-generate pipeline
2. **Advanced RAG**: Pre-retrieval processing, optimized indexing, post-processing
3. **Modular RAG**: Flexible module composition, iterative retrieval, dynamic routing

**RAG Components:**
- **Retrieval**: Dense/sparse retrieval, hybrid methods, query rewriting
- **Generation**: Context integration, prompt engineering, answer synthesis
- **Augmentation**: Chunking strategies, embedding optimization, re-ranking

**Relevance to Intent Clustering:**
- Query understanding in RAG systems aligns with intent recognition
- Retrieval results can inform intent refinement
- However, existing RAG systems lack explicit intent modeling in retrieval

### 4.2 Knowledge Graph Enhanced RAG

Recent work integrates knowledge graphs with RAG:
- **GraphRAG**: Uses knowledge graph for community summarization
- **KG-RAG**: Combines entity linking with retrieval
- **ToG (Think-on-Graph)**: Reasoning paths through KG

**Research Gap:** Current KG-RAG systems retrieve relevant subgraphs but don't model user intent as explicit nodes or use intent for query reformulation.

---

## 5. Comparative Analysis

### 5.1 Related Work Comparison Table

| Approach | Intent Modeling | Knowledge Graph | User Feedback | Dynamic Weight | Clustering |
|----------|-----------------|-----------------|---------------|----------------|------------|
| **IKGR** (2025) | ✅ LLM-extracted | ✅ Intent-centric | ❌ | ❌ | ❌ |
| **IKG-EMR** (2025) | ✅ Topic-based | ✅ Tripartite | ❌ | ✅ Adaptive fusion | ❌ |
| **MIND** (2019) | ⚠️ Multi-interest | ❌ | ❌ | ⚠️ Label-aware | ✅ Routing |
| **KGAT** (2019) | ❌ | ✅ Attention-based | ❌ | ⚠️ Attention | ❌ |
| **RAG Systems** | ⚠️ Query-level | ⚠️ Optional | ❌ | ❌ | ❌ |
| **Proposed Approach** | ✅ Explicit | ✅ Intent nodes | ✅ Real-time | ✅ Dynamic | ✅ Unsupervised |

### 5.2 Methodology Taxonomy

```
Intent-Aware Systems
├── Implicit Intent
│   ├── Behavior-based (SR-GNN, DSSRec)
│   └── Topic-based (IKG-EMR)
├── Explicit Intent
│   ├── LLM-extracted (IKGR)
│   └── User-declared (limited work)
└── Hybrid Approaches
    └── Proposed: LLM + User Feedback + Clustering

Knowledge Graph Integration
├── Auxiliary Information (KGCN, KGAT)
├── Core Structure (IKG-EMR, IKGR)
└── Intent-Centric (Proposed)
```

---

## 6. Research Gaps and Opportunities

### 6.1 Identified Gaps

1. **Lack of User Feedback Loop**
   - IKGR extracts intents offline, no refinement mechanism
   - IKG-EMR uses static topic-based intent, cannot adapt

2. **Static Intent Representation**
   - Once intent is determined, no dynamic adjustment
   - User intent evolves over time, not captured

3. **No Unsupervised Intent Clustering in KG**
   - MIND clusters interests but without KG structure
   - KG methods don't cluster intent patterns across users

4. **Missing Intent-Aware Retrieval in RAG**
   - RAG systems use query similarity, not intent matching
   - No intent-based query reformulation

5. **Limited Intent Hierarchy**
   - Most methods model single-level intent
   - No hierarchical intent representation (goal → sub-goals → actions)

### 6.2 Opportunities for Proposed Approach

| Gap | Proposed Solution |
|-----|-------------------|
| No user feedback loop | **ReAct Agent** for interactive intent refinement |
| Static intent | **Dynamic weight adjustment** based on feedback |
| No KG clustering | **Unsupervised intent clustering** in embedding space |
| Missing intent retrieval | Intent nodes as first-class entities in KG-RAG |

---

## 7. Contradictory Findings

### 7.1 Intent Granularity Trade-off

- **IKG-EMR** argues for **coarse-grained intent** (topics) to reduce sparsity
- **MIND** demonstrates **fine-grained multi-interest** improves accuracy
- **Resolution**: Hierarchical intent model needed (coarse for cold-start, fine for engagement)

### 7.2 LLM vs. Traditional Methods

- **IKGR** claims LLM-based intent extraction outperforms KGAT
- **IKG-EMR** achieves strong results with traditional GNN + Topic modeling
- **Resolution**: Hybrid approach - LLM for complex intents, traditional for well-defined domains

### 7.3 Graph Construction

- **IKG-EMR**: User-Item-Topic tripartite graph
- **IKGR**: User-Intent-Item with mutual-intent connectivity
- **Resolution**: Intent nodes should connect both users and items, with learned relationships

---

## 8. Recommendations for Proposed Research

Based on this literature review, the proposed research should:

1. **Build on IKGR's intent-centric graph** but add:
   - ReAct Agent for interactive intent refinement
   - User feedback mechanism for weight adjustment
   - Unsupervised clustering for intent pattern discovery

2. **Adopt IKG-EMR's tripartite structure** with modifications:
   - Replace LDA topics with learned intent embeddings
   - Add intent-intent relationships
   - Enable real-time weight updates

3. **Leverage MIND's multi-interest routing** adapted for:
   - Intent rather than interest
   - Knowledge graph structure
   - Dynamic routing weights

4. **Integrate with RAG systems** via:
   - Intent-aware retrieval
   - Intent-based query reformulation
   - Contextual intent refinement through generation

---

## 9. Key References

1. Zheng, W. et al. (2025). "Tuning-Free LLM Can Build A Strong Recommender Under Sparse Connectivity And Knowledge Gap Via Extracting Intent." arXiv:2505.10900.

2. Intent-aware knowledge graph-based model for electrical power material recommendation. PeerJ Computer Science, 2025.

3. Li, C. et al. (2019). "Multi-Interest Network with Dynamic Routing for Recommendation at Tmall." CIKM 2019.

4. Wang, X. et al. (2019). "KGAT: Knowledge Graph Attention Network for Recommendation." KDD 2019.

5. Gao, Y. et al. (2024). "Retrieval-Augmented Generation for Large Language Models: A Survey." arXiv:2312.10997.

---

## Appendix: Additional Relevant Papers

- Wang, X. et al. (2020). "CKAN: Collaborative Knowledge-aware Attentive Network for Recommendation." MM 2020.
- Wang, H. et al. (2019). "KGCN: Knowledge Graph Convolutional Networks for Recommender Systems." WWW 2019.
- Wu, S. et al. (2019). "Session-based Recommendation with Graph Neural Networks." AAAI 2019.
- Ma, J. et al. (2020). "DSSRec: Disentangled Self-Supervision in Sequential Recommender Systems." KDD 2020.

---

*End of Literature Review*