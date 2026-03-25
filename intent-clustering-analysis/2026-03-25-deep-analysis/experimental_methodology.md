# Experimental Methodology

> **Section**: Methodology  
> **Author**: methodology-designer  
> **Date**: 2026-03-25  
> **Team**: intent-clustering-team

---

## 1. Research Questions

This study addresses the following research questions:

| RQ | Description |
|----|-------------|
| **RQ1** | How does unsupervised intent clustering improve retrieval accuracy in knowledge graph-based RAG systems? |
| **RQ2** | What is the impact of dynamic weight mechanisms on user satisfaction and retrieval relevance? |
| **RQ3** | How effective is the proposed RLHF framework for adapting retrieval strategies to user feedback? |
| **RQ4** | What are the relative contributions of different components (clustering, weighting, RLHF) to overall system performance? |

---

## 2. Experimental Setup

### 2.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Experimental System Architecture                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │                    ReAct Agent Framework                      │  │
│   │   ┌────────────┐  ┌────────────┐  ┌────────────┐            │  │
│   │   │  Reasoning │  │   Action   │  │ Observation│            │  │
│   │   │   Module   │──►│  Selector  │──►│  Parser   │            │  │
│   │   └────────────┘  └────────────┘  └────────────┘            │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │              Intent-Aware Retrieval Module                    │  │
│   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │  │
│   │   │   Intent    │  │   Dynamic   │  │   Graph     │         │  │
│   │   │  Clustering │  │   Weights   │  │   Traversal │         │  │
│   │   └─────────────┘  └─────────────┘  └─────────────┘         │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │                    Knowledge Graph Store                       │  │
│   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │  │
│   │   │   Entities  │  │  Relations  │  │  Documents  │         │  │
│   │   └─────────────┘  └─────────────┘  └─────────────┘         │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │                   RLHF Feedback Loop                          │  │
│   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │  │
│   │   │   Reward    │  │   Policy    │  │   Weight    │         │  │
│   │   │   Model     │  │   Network   │  │   Updater   │         │  │
│   │   └─────────────┘  └─────────────┘  └─────────────┘         │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Hardware and Software Environment

| Component | Specification |
|-----------|---------------|
| **GPU** | NVIDIA A100 80GB × 4 (for embedding and LLM inference) |
| **CPU** | AMD EPYC 7742 64-core (for clustering and graph traversal) |
| **Memory** | 512GB RAM |
| **Storage** | 4TB NVMe SSD |
| **Framework** | PyTorch 2.1, Transformers 4.36, NetworkX 3.2 |
| **Graph DB** | Neo4j 5.15 (for knowledge graph storage) |
| **Vector DB** | FAISS 1.7 (for embedding index) |

---

## 3. Datasets

### 3.1 Primary Datasets

#### Dataset 1: MS MARCO Question-Answering Dataset

| Property | Value |
|----------|-------|
| **Size** | 1,010,916 queries |
| **Type** | Real user questions from Bing |
| **Annotations** | Human-annotated relevance labels |
| **Split** | Train: 80%, Val: 10%, Test: 10% |
| **Purpose** | Intent clustering validation |

#### Dataset 2: Natural Questions (NQ)

| Property | Value |
|----------|-------|
| **Size** | 307,373 training examples |
| **Type** | Natural questions with Wikipedia answers |
| **Annotations** | Long and short answer spans |
| **Purpose** | Knowledge graph construction |

#### Dataset 3: WebQuestions (WebQ)

| Property | Value |
|----------|-------|
| **Size** | 5,810 questions |
| **Type** | Freebase KG-based questions |
| **Purpose** | Graph traversal evaluation |

#### Dataset 4: Custom User Interaction Logs

| Property | Value |
|----------|-------|
| **Size** | 100,000 interaction sessions |
| **Type** | User queries + feedback signals |
| **Feedback Types** | Explicit (like/dislike), Implicit (dwell time, clicks) |
| **Purpose** | RLHF training and evaluation |

### 3.2 Knowledge Graph Construction

```python
def construct_knowledge_graph(documents, entities, relations):
    """
    Construct knowledge graph from documents.
    
    Args:
        documents: List of text documents
        entities: Pre-extracted entity list
        relations: Pre-defined relation types
    
    Returns:
        G: NetworkX MultiDiGraph
    """
    G = nx.MultiDiGraph()
    
    for doc in documents:
        # Extract entities using spaCy or custom NER
        doc_entities = extract_entities(doc)
        
        # Add entity nodes
        for entity in doc_entities:
            G.add_node(entity['id'], 
                      type=entity['type'],
                      text=entity['text'],
                      doc_id=doc['id'])
        
        # Extract relations using pattern matching or RE model
        doc_relations = extract_relations(doc, doc_entities)
        
        # Add relation edges
        for rel in doc_relations:
            G.add_edge(rel['head'], rel['tail'],
                      relation=rel['type'],
                      weight=rel['confidence'])
    
    # Compute node embeddings
    node_embeddings = compute_node_embeddings(G, method='node2vec')
    
    return G, node_embeddings
```

### 3.3 Dataset Statistics

| Dataset | Queries | Entities | Relations | Avg. Query Length |
|---------|---------|----------|-----------|-------------------|
| MS MARCO | 1,010,916 | 2.1M | 5.8M | 6.3 tokens |
| Natural Questions | 307,373 | 1.8M | 4.2M | 9.1 tokens |
| WebQuestions | 5,810 | 0.4M | 1.1M | 7.2 tokens |
| Custom Logs | 100,000 | - | - | 8.5 tokens |

---

## 4. Baseline Methods

### 4.1 Retrieval Baselines

| Method | Description | Reference |
|--------|-------------|-----------|
| **BM25** | Traditional lexical retrieval | Robertson & Zaragoza, 2009 |
| **Dense Retrieval (DPR)** | Dual-encoder dense retrieval | Karpukhin et al., 2020 |
| **ColBERT** | Late interaction retrieval | Khattab & Zaharia, 2020 |
| **RAPTOR** | Recursive summarization for retrieval | Sarthi et al., 2024 |
| **GraphRAG** | Graph-enhanced RAG | Edge et al., 2024 |
| **KGR (Knowledge Graph Retrieval)** | Pure KG-based retrieval | Zhang et al., 2022 |

### 4.2 Clustering Baselines

| Method | Description |
|--------|-------------|
| **K-Means** | Standard K-Means on query embeddings |
| **DBSCAN** | Density-based clustering |
| **Agglomerative** | Hierarchical agglomerative clustering |
| **GMM** | Gaussian Mixture Model |
| **BERT-KMeans** | K-Means on BERT [CLS] embeddings |

### 4.3 RLHF Baselines

| Method | Description |
|--------|-------------|
| **No-RLHF** | Static weights, no feedback learning |
| **Simple Bandit** | Multi-armed bandit for weight selection |
| **REINFORCE** | Policy gradient method |
| **DPO** | Direct Preference Optimization |

---

## 5. Evaluation Metrics

### 5.1 Retrieval Quality Metrics

```python
def compute_retrieval_metrics(predicted_docs, relevant_docs, k_list=[1, 5, 10, 20]):
    """
    Compute standard retrieval metrics.
    
    Args:
        predicted_docs: List of predicted document IDs (ranked)
        relevant_docs: Set of relevant document IDs (ground truth)
        k_list: List of cutoff values for @K metrics
    
    Returns:
        metrics: Dict of metric names to values
    """
    metrics = {}
    
    # Precision@K
    for k in k_list:
        hits = len(set(predicted_docs[:k]) & relevant_docs)
        metrics[f'P@{k}'] = hits / k
    
    # Recall@K
    for k in k_list:
        hits = len(set(predicted_docs[:k]) & relevant_docs)
        metrics[f'R@{k}'] = hits / len(relevant_docs) if relevant_docs else 0
    
    # MRR (Mean Reciprocal Rank)
    for i, doc in enumerate(predicted_docs):
        if doc in relevant_docs:
            metrics['MRR'] = 1.0 / (i + 1)
            break
    else:
        metrics['MRR'] = 0
    
    # NDCG@K
    for k in k_list:
        dcg = sum(
            1.0 / np.log2(i + 2) 
            for i, doc in enumerate(predicted_docs[:k]) 
            if doc in relevant_docs
        )
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_docs))))
        metrics[f'NDCG@{k}'] = dcg / idcg if idcg > 0 else 0
    
    # MAP (Mean Average Precision)
    num_relevant = 0
    precision_sum = 0
    for i, doc in enumerate(predicted_docs):
        if doc in relevant_docs:
            num_relevant += 1
            precision_sum += num_relevant / (i + 1)
    metrics['MAP'] = precision_sum / len(relevant_docs) if relevant_docs else 0
    
    return metrics
```

### 5.2 Clustering Quality Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Silhouette Score** | $s = \frac{b-a}{\max(a,b)}$ | Range [-1, 1], higher is better |
| **Calinski-Harabasz Index** | $\frac{SS_B}{SS_W} \times \frac{n-k}{k-1}$ | Higher is better |
| **Davies-Bouldin Index** | $\frac{1}{k}\sum_{i=1}^{k}\max_{j\neq i}\frac{\sigma_i+\sigma_j}{d(c_i,c_j)}$ | Lower is better |
| **Adjusted Rand Index (ARI)** | $\frac{RI - E[RI]}{\max(RI) - E[RI]}$ | Range [-1, 1], 1 is perfect |

### 5.3 User Satisfaction Metrics

```python
def compute_satisfaction_metrics(feedback_log):
    """
    Compute user satisfaction metrics from feedback logs.
    
    Args:
        feedback_log: List of FeedbackEvent objects
    
    Returns:
        metrics: Dict of satisfaction metrics
    """
    metrics = {}
    
    # Explicit satisfaction rate
    likes = sum(1 for fb in feedback_log if fb.explicit_feedback.get('like', False))
    dislikes = sum(1 for fb in feedback_log if fb.explicit_feedback.get('dislike', False))
    metrics['satisfaction_rate'] = likes / (likes + dislikes) if (likes + dislikes) > 0 else 0
    
    # Click-through rate (CTR)
    clicks = sum(1 for fb in feedback_log if fb.clicked_result)
    metrics['CTR'] = clicks / len(feedback_log)
    
    # Average dwell time
    metrics['avg_dwell_time'] = np.mean([fb.dwell_time for fb in feedback_log])
    
    # Task completion rate
    completed = sum(1 for fb in feedback_log if fb.task_completed)
    metrics['completion_rate'] = completed / len(feedback_log)
    
    # Net Promoter Score (NPS) proxy
    promoters = sum(1 for fb in feedback_log if fb.rating and fb.rating >= 4)
    detractors = sum(1 for fb in feedback_log if fb.rating and fb.rating <= 2)
    metrics['NPS'] = (promoters - detractors) / len(feedback_log) * 100
    
    return metrics
```

### 5.4 System Efficiency Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| **Latency (P50)** | < 200ms | Median response time |
| **Latency (P99)** | < 500ms | 99th percentile response time |
| **Throughput** | > 100 QPS | Queries per second |
| **Memory Usage** | < 64GB | Peak memory consumption |
| **Clustering Time** | < 30s | Time for full recluster |

---

## 6. Main Experiments

### 6.1 Experiment 1: Intent Clustering Effectiveness (RQ1)

**Objective**: Evaluate the quality and stability of unsupervised intent clustering.

**Procedure**:

```
Algorithm 1: Intent Clustering Evaluation
─────────────────────────────────────────
Input: Query set Q = {q₁, q₂, ..., qₙ}, Ground truth intents I*
Output: Clustering quality metrics

1: for each embedding model E ∈ {BERT, RoBERTa, SimCSE, E5} do
2:     Embeddings ← ComputeEmbeddings(Q, E)
3:     
4:     for each clustering algorithm A ∈ {K-Means, DBSCAN, HDBSCAN, GMM} do
5:         Clusters ← Cluster(Embeddings, A)
6:         
7:         // Internal metrics (no ground truth)
8:         silhouette ← ComputeSilhouette(Embeddings, Clusters)
9:         calinski_harabasz ← ComputeCH(Embeddings, Clusters)
10:        
11:        // External metrics (with ground truth)
12:        ari ← ComputeARI(Clusters, I*)
13:        nmi ← ComputeNMI(Clusters, I*)
14:        
15:        // Stability test
16:        for i = 1 to 5 do
17:            Clusters_i ← Cluster(Embeddings[shuffle], A)
18:            stability_scores.append(ComputeARI(Clusters, Clusters_i))
19:        end for
20:        
21:        Report(A, E, silhouette, calinski_harabasz, ari, nmi, mean(stability_scores))
22:    end for
23: end for
```

**Metrics**: Silhouette Score, Calinski-Harabasz Index, ARI, NMI, Cluster Stability

### 6.2 Experiment 2: Retrieval Performance (RQ2)

**Objective**: Compare retrieval performance with and without intent-aware mechanisms.

**Procedure**:

```
Algorithm 2: Retrieval Performance Evaluation
──────────────────────────────────────────────
Input: Queries Q, Knowledge Graph G, Ground truth relevance R*
Output: Retrieval metrics comparison

1: // Baseline: Standard semantic search
2: Baseline_results ← {}
3: for q in Q do
4:     embedding ← Encode(q)
5:     candidates ← VectorSearch(embedding, k=100)
6:     Baseline_results[q] ← candidates
7: end for

8: // Proposed: Intent-aware retrieval
9: Proposed_results ← {}
10: Clusters ← IntentCluster(Q)  // Pre-computed clusters
11: Weights ← InitializeWeights(G, Clusters)
12:
13: for q in Q do
14:     embedding ← Encode(q)
15:     cluster ← AssignCluster(q, Clusters)
16:     
17:     // Dual-path recall
18:     semantic_candidates ← VectorSearch(embedding, k=50)
19:     intent_candidates ← GraphTraversal(G, cluster, Weights, k=50)
20:     
21:     // Fusion
22:     final_candidates ← Fusion(semantic_candidates, intent_candidates, α=0.6)
23:     Proposed_results[q] ← final_candidates
24: end for

25: // Compute metrics
26: for method in {Baseline, Proposed} do
27:     for k in {1, 5, 10, 20} do
28:         Report(method, P@k, R@k, NDCG@k, MRR)
29:     end for
30: end for
```

### 6.3 Experiment 3: RLHF Adaptation (RQ3)

**Objective**: Evaluate the effectiveness of RLHF-based weight adaptation.

**Procedure**:

```
Algorithm 3: RLHF Adaptation Experiment
───────────────────────────────────────
Input: Query set Q, Initial weights W₀, Feedback log F
Output: Weight adaptation metrics

1: // Split users into groups
2: Users_A, Users_B, Users_C ← SplitUsers(users, ratio=[0.4, 0.3, 0.3])

3: // Group A: No RLHF (static weights)
4: W_A ← W₀  // Fixed weights

5: // Group B: Simple bandit
6: W_B ← W₀
7: for feedback in F[Users_B] do
8:     W_B ← BanditUpdate(W_B, feedback)
9: end for

10: // Group C: PPO-based RLHF
11: policy ← InitializePolicy(W₀)
12: for episode in training_episodes do
13:     batch ← SampleBatch(F[Users_C])
14:     rewards ← ComputeRewards(batch)
15:     policy ← PPOUpdate(policy, batch, rewards)
16: end for
17: W_C ← ExtractWeights(policy)

18: // A/B test evaluation
19: for group, weights in {(Users_A, W_A), (Users_B, W_B), (Users_C, W_C)} do
20:     satisfaction_rate ← EvaluateSatisfaction(group, weights)
21:     convergence_speed ← MeasureConvergence(group)
22:     Report(group, satisfaction_rate, convergence_speed)
23: end for
```

---

## 7. Ablation Studies

### 7.1 Ablation Study Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    Ablation Study Matrix                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Full Model = Intent Clustering + Dynamic Weights + RLHF       │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Config │ Clustering │ Dynamic Weights │ RLHF │ Metrics │   │
│   ├─────────────────────────────────────────────────────────┤   │
│   │  Full   │     ✓      │       ✓         │  ✓   │ Baseline│   │
│   │  -RLHF  │     ✓      │       ✓         │  ✗   │ Δ₁      │   │
│   │  -W     │     ✓      │       ✗         │  ✓   │ Δ₂      │   │
│   │  -C     │     ✗      │       ✓         │  ✓   │ Δ₃      │   │
│   │  -C-W   │     ✗      │       ✗         │  ✓   │ Δ₄      │   │
│   │  -All   │     ✗      │       ✗         │  ✗   │ Δ₅      │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   Contribution of each component:                               │
│   - Clustering contribution: Δ₃ - Δ₅                           │
│   - Dynamic Weights contribution: Δ₂ - Δ₅                      │
│   - RLHF contribution: Δ₁ - Δ₅                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Component-wise Ablation

#### Ablation 1: Intent Clustering Impact

```python
def ablation_clustering(queries, knowledge_graph, use_clustering=True):
    """
    Ablation study for intent clustering component.
    """
    if use_clustering:
        # Full model with clustering
        clusters = hdbscan_clustering(query_embeddings)
        cluster_assignments = assign_to_clusters(queries, clusters)
        recall_results = intent_aware_recall(queries, cluster_assignments, knowledge_graph)
    else:
        # Ablated: direct semantic search without clustering
        recall_results = semantic_recall(queries, knowledge_graph)
    
    return compute_metrics(recall_results, ground_truth)
```

**Hypothesis**: Intent clustering provides 5-15% improvement in retrieval accuracy.

#### Ablation 2: Dynamic Weight Impact

```python
def ablation_weights(queries, clusters, knowledge_graph, use_dynamic=True):
    """
    Ablation study for dynamic weight component.
    """
    if use_dynamic:
        # Full model with dynamic weights
        weights = compute_dynamic_weights(clusters, feedback_history)
        recall_results = weighted_recall(queries, clusters, knowledge_graph, weights)
    else:
        # Ablated: uniform weights
        weights = uniform_weights(knowledge_graph)
        recall_results = weighted_recall(queries, clusters, knowledge_graph, weights)
    
    return compute_metrics(recall_results, ground_truth)
```

**Hypothesis**: Dynamic weights provide 3-10% improvement in user satisfaction.

#### Ablation 3: RLHF Impact

```python
def ablation_rlhf(queries, initial_weights, feedback_log, use_rlhf=True):
    """
    Ablation study for RLHF component.
    """
    if use_rlhf:
        # Full model with RLHF
        trained_policy = ppo_train(initial_policy, feedback_log, epochs=100)
        adapted_weights = extract_weights(trained_policy)
    else:
        # Ablated: static weights
        adapted_weights = initial_weights
    
    return evaluate_weights(adapted_weights, test_queries)
```

**Hypothesis**: RLHF provides 8-20% improvement in long-term user satisfaction.

### 7.3 Hyperparameter Sensitivity Analysis

| Hyperparameter | Range | Impact Metric |
|---------------|-------|---------------|
| Clustering `min_cluster_size` | [5, 10, 20, 50] | ARI, NMI |
| Weight fusion α | [0.3, 0.5, 0.7, 0.9] | NDCG@10 |
| PPO clip ratio ε | [0.1, 0.2, 0.3] | Satisfaction rate |
| Learning rate η | [1e-5, 1e-4, 1e-3] | Convergence speed |
| Reward decay λ | [0.9, 0.95, 0.99] | Long-term reward |

---

## 8. Statistical Analysis

### 8.1 Significance Testing

```python
from scipy import stats

def statistical_significance_test(baseline_scores, treatment_scores, alpha=0.05):
    """
    Perform statistical significance testing.
    
    Returns:
        p_value, effect_size, is_significant
    """
    # Paired t-test (for paired samples)
    t_statistic, p_value = stats.ttest_rel(baseline_scores, treatment_scores)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        (np.std(baseline_scores)**2 + np.std(treatment_scores)**2) / 2
    )
    cohens_d = (np.mean(treatment_scores) - np.mean(baseline_scores)) / pooled_std
    
    # Bonferroni correction for multiple comparisons
    corrected_alpha = alpha / num_comparisons
    is_significant = p_value < corrected_alpha
    
    return p_value, cohens_d, is_significant
```

### 8.2 Confidence Intervals

```python
def compute_confidence_interval(scores, confidence=0.95):
    """Compute bootstrap confidence interval."""
    n_bootstrap = 10000
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    lower = np.percentile(bootstrap_means, (1 - confidence) / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 + confidence) / 2 * 100)
    
    return lower, upper
```

---

## 9. Reproducibility

### 9.1 Random Seed Configuration

```python
import random
import numpy as np
import torch

def set_random_seeds(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 9.2 Experiment Logging

```python
import mlflow

def log_experiment(params, metrics, artifacts):
    """Log experiment details for reproducibility."""
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model artifacts
        for name, path in artifacts.items():
            mlflow.log_artifact(path, artifact_path=name)
        
        # Log code version
        mlflow.set_tag("git_commit", get_git_commit_hash())
        mlflow.set_tag("python_version", sys.version)
```

### 9.3 Configuration File

```yaml
# config/experiment_config.yaml
experiment:
  name: "intent_clustering_rag"
  seed: 42
  num_runs: 5

data:
  dataset: "msmarco"
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1

model:
  embedding: "sentence-transformers/all-mpnet-base-v2"
  clustering:
    algorithm: "hdbscan"
    min_cluster_size: 10
    min_samples: 5
  weights:
    init_method: "uniform"
    learning_rate: 0.01
    momentum: 0.9

training:
  epochs: 100
  batch_size: 256
  optimizer: "adamw"
  lr: 1e-4
  
rlhf:
  algorithm: "ppo"
  clip_ratio: 0.2
  value_coef: 0.5
  entropy_coef: 0.01

evaluation:
  metrics: ["ndcg@10", "mrr", "map", "satisfaction_rate"]
  k_values: [1, 5, 10, 20]
```

---

## 10. Expected Results

### 10.1 Anticipated Performance Gains

| Component | Expected Improvement | Confidence |
|-----------|---------------------|------------|
| Intent Clustering | +8-12% NDCG@10 | High |
| Dynamic Weights | +5-8% Satisfaction | Medium |
| RLHF Adaptation | +10-15% Long-term Satisfaction | Medium |
| Full System | +20-30% Overall | Medium-High |

### 10.2 Result Tables (Template)

**Table 1: Retrieval Performance Comparison**

| Method | P@10 | R@10 | NDCG@10 | MRR | MAP |
|--------|------|------|---------|-----|-----|
| BM25 | - | - | - | - | - |
| DPR | - | - | - | - | - |
| GraphRAG | - | - | - | - | - |
| **Ours (Full)** | - | - | - | - | - |

**Table 2: Ablation Results**

| Configuration | NDCG@10 | Δ vs Full | Satisfaction Rate |
|---------------|---------|-----------|-------------------|
| Full Model | - | - | - |
| - RLHF | - | -% | - |
| - Dynamic Weights | - | -% | - |
| - Clustering | - | -% | - |
| Baseline (no components) | - | -% | - |

---

## 11. Limitations and Threats to Validity

### 11.1 Internal Validity

| Threat | Mitigation |
|--------|------------|
| Implementation bugs | Unit tests, code review, reference implementations |
| Hyperparameter tuning | Grid search with cross-validation |
| Random initialization | Multiple runs with different seeds |

### 11.2 External Validity

| Threat | Mitigation |
|--------|------------|
| Dataset bias | Multiple datasets from different domains |
| User behavior variance | Diverse user groups in feedback collection |
| Temporal drift | Longitudinal evaluation over 6+ months |

### 11.3 Construct Validity

| Threat | Mitigation |
|--------|------------|
| Metric alignment | Multiple complementary metrics |
| User satisfaction proxy | Direct user surveys alongside implicit signals |
| Intent ground truth | Expert annotation with inter-rater agreement |

---

## 12. Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| **Data Preparation** | 2 weeks | Dataset collection, KG construction |
| **Implementation** | 4 weeks | Core system development |
| **Baseline Experiments** | 2 weeks | Baseline reproduction |
| **Main Experiments** | 3 weeks | RQ1-RQ4 evaluation |
| **Ablation Studies** | 2 weeks | Component analysis |
| **Analysis & Writing** | 3 weeks | Results analysis, paper writing |
| **Total** | 16 weeks | - |

---

## Appendix A: Pseudocode for Core Algorithms

### A.1 Intent-Aware Retrieval Algorithm

```
Algorithm: IntentAwareRetrieval
────────────────────────────────
Input: Query q, Knowledge Graph G, Cluster Model C, Weight Matrix W
Output: Ranked list of relevant nodes

1.  q_emb ← ENCODE(q)                           // Query embedding
2.  cluster ← ASSIGN_CLUSTER(q_emb, C)          // Intent assignment
3.  cluster_prob ← SOFT_ASSIGNMENT(q_emb, C)    // Probability distribution

4.  // Semantic recall path
5.  semantic_candidates ← VECTOR_SEARCH(q_emb, k=50)
6.  semantic_scores ← COSINE_SIM(q_emb, semantic_candidates)

7.  // Intent-based recall path
8.  seed_nodes ← GET_SEED_NODES(cluster, G)
9.  FOR each seed_node IN seed_nodes DO
10.     traversed ← GRAPH_TRAVERSAL(G, seed_node, max_depth=3)
11.     FOR each node IN traversed DO
12.         intent_weight ← W[node][cluster]
13.         path_confidence ← COMPUTE_PATH_CONFIDENCE(node, seed_node)
14.         intent_scores[node] ← intent_weight * path_confidence
15.     END FOR
16. END FOR

17. // Fusion
18. FOR each candidate IN ALL_CANDIDATES DO
19.     semantic_score ← semantic_scores.get(candidate, 0)
20.     intent_score ← intent_scores.get(candidate, 0)
21.     final_score[candidate] ← α * semantic_score + (1-α) * intent_score
22. END FOR

23. RETURN TOP_K(final_score, k=20)
```

### A.2 Dynamic Weight Update Algorithm

```
Algorithm: DynamicWeightUpdate
──────────────────────────────
Input: Current weights W, Feedback event f, Learning rate η
Output: Updated weights W'

1.  reward ← COMPUTE_REWARD(f)                   // From reward function
2.  node ← f.recalled_node
3.  cluster ← f.intent_cluster

4.  // Momentum-based update
5.  velocity[node][cluster] ← β * velocity[node][cluster] + η * reward
6.  W[node][cluster] ← W[node][cluster] + velocity[node][cluster]

7.  // Softmax normalization
8.  FOR each cluster_id IN CLUSTERS DO
9.     exp_weights[cluster_id] ← exp(W[node][cluster_id])
10. END FOR
11. total ← SUM(exp_weights)
12. FOR each cluster_id IN CLUSTERS DO
13.    W'[node][cluster_id] ← exp_weights[cluster_id] / total
14. END FOR

15. RETURN W'
```

### A.3 PPO Training for Weight Adaptation

```
Algorithm: PPOWeightTraining
────────────────────────────
Input: Initial policy π_θ, Feedback dataset D, Epochs E, Clip ratio ε
Output: Trained policy π_θ*

1.  FOR epoch = 1 TO E DO
2.     batch ← SAMPLE_BATCH(D, size=B)
3.     
4.     FOR each (state, action, reward, next_state) IN batch DO
5.         // Compute advantage
6.         V(s) ← VALUE_NETWORK(state)
7.         V(s') ← VALUE_NETWORK(next_state)
8.         A ← reward + γ * V(s') - V(s)
9.         
10.        // Compute probability ratio
11.        π_old ← POLICY_NETWORK_old(state)[action]
12.        π_new ← POLICY_NETWORK(state)[action]
13.        ratio ← π_new / π_old
14.        
15.        // Clipped objective
16.        L_clip ← MIN(ratio * A, CLIP(ratio, 1-ε, 1+ε) * A)
17.        
18.        // Value loss
19.        L_value ← MSE(V(s), reward + γ * V(s'))
20.        
21.        // Total loss
22.        L_total ← -L_clip + c1 * L_value - c2 * ENTROPY(π)
23.        
24.        // Update networks
25.        θ ← θ - α * ∇_θ L_total
26.    END FOR
27. END FOR

28. RETURN π_θ
```

---

## Appendix B: Data Processing Pipeline

```python
class DataProcessingPipeline:
    """Complete data processing pipeline for the experiment."""
    
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.embedding_model)
        self.encoder = AutoModel.from_pretrained(config.embedding_model)
        
    def process_queries(self, query_file):
        """Process raw queries into experiment-ready format."""
        # Load
        queries = self._load_queries(query_file)
        
        # Clean
        queries = self._clean_queries(queries)
        
        # Tokenize
        tokenized = self.tokenizer(
            queries, 
            padding=True, 
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Embed
        with torch.no_grad():
            embeddings = self.encoder(**tokenized).last_hidden_state[:, 0, :]
        
        # Entity extraction
        entities = self._extract_entities(queries)
        
        return {
            'queries': queries,
            'embeddings': embeddings.numpy(),
            'entities': entities,
            'tokenized': tokenized
        }
    
    def _clean_queries(self, queries):
        """Clean and normalize queries."""
        import re
        cleaned = []
        for q in queries:
            # Lowercase
            q = q.lower()
            # Remove special characters
            q = re.sub(r'[^\w\s\?]', '', q)
            # Normalize whitespace
            q = ' '.join(q.split())
            cleaned.append(q)
        return cleaned
    
    def _extract_entities(self, queries):
        """Extract named entities from queries."""
        import spacy
        nlp = spacy.load('en_core_web_sm')
        
        entities_list = []
        for q in queries:
            doc = nlp(q)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            entities_list.append(entities)
        
        return entities_list
```

---

*Document Version: v1.0 | Created: 2026-03-25 | Team: intent-clustering-team*