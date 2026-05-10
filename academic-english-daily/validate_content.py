#!/usr/bin/env python3
"""Validate content.json against avoid_* lists."""
import json

data = json.loads(open("content.json").read())

# Vocabulary avoid list
avoid_vocab = [
    "attend to", "convex", "negligible", "orthogonal", "elucidate",
    "by virtue of", "eigenspace", "circumvent", "conversely", "curvature",
    "amortize", "prominent", "isomorphic", "delineate", "subspace",
    "aggregate", "invariant", "probe", "surrogate", "divergence",
    "monotonic", "retrieval", "ground", "fidelity", "rerank", "chunk",
    "encode", "autoregressive", "causal", "residual", "receptive field",
    "downsample", "tokenization", "line search", "Lipschitz", "saddle point",
    "spectral", "postulate", "decompose", "Hessian", "mutual information",
    "partition function", "calibration", "ablation", "bootstrap",
    "penalize", "shrinkage", "sparse"
]

print("=== Vocabulary Check ===")
for w in data["vocabulary"]:
    word = w["word"]
    ok = word not in avoid_vocab
    print(f"  {'PASS' if ok else 'FAIL'}: {word}")
    if not ok:
        raise SystemExit(1)

# Patterns avoid list
avoid_patterns = [
    "We empirically demonstrate that [method] outperforms [baseline] by [metric] on [benchmark].",
    "A key insight of our approach is that [X] can be reinterpreted as [Y], enabling [benefit].",
    "While [X] is well-established for [task], it suffers from [limitation], particularly when [condition].",
    "The computational cost of [operation] scales [quadratically/linearly] with [variable], rendering it impractical for [scenario].",
    "We hypothesize that [phenomenon] arises from [cause], which is corroborated by our analysis in Section [N].",
    "[Technique] serves as a drop-in replacement for [standard approach], requiring minimal architectural changes while providing [benefit].",
    "The performance gap widens as [variable] increases, suggesting that [method] is particularly sensitive to [factor].",
    "Rather than [heuristic approach], we formulate the problem as a [structured formulation], enabling [desirable property].",
    "We constrain [quantity] to lie within a [region] by [mechanism], thereby ensuring [property].",
    "Although [approach A] and [approach B] share the same [objective], they diverge in how they [aspect].",
    "Empirical results across [N] tasks corroborate that [finding], with a mean [metric] of [value].",
    "By coupling [component A] with [component B], our pipeline simultaneously achieves [property X] and [property Y], a synergy that neither subsystem realizes in isolation.",
    "We decompose [task] into two complementary subproblems: [subtask A], which governs [aspect A], and [subtask B], which handles [aspect B].",
    "[Method] operates along a continuum between [extreme A] and [extreme B], with [parameter] governing the inherent trade-off.",
    "Beyond [standard approach], our framework incorporates [additional mechanism] to [benefit], which proves especially effective under [condition].",
    "The interplay between [component A] and [component B] gives rise to [phenomenon], highlighting the importance of [insight].",
    "We characterize the conditions under which [method] succeeds versus [alternative], revealing that [factor] is the primary determinant of [outcome].",
    "To [address/mitigate] [issue], we incorporate [mechanism] into the [component], which [effect] while preserving [desired property].",
    "The [architecture] comprises [N] stacked [component]s, each of which interleaves [sublayer A] with [sublayer B], followed by [sublayer C] with [operation] applied after every [unit].",
    "A distinguishing feature of our design is the adoption of [technique] in lieu of [standard practice], which yields [benefit] at the expense of [cost].",
    "The distinguishing characteristic of [method A] relative to [method B] is its reliance on [inductive bias], which [effect] under [condition].",
    "[Operation] can be interpreted as [mathematical abstraction] over the [domain], where each [element] interacts with its [neighborhood] via [mechanism].",
    "A common strategy to [goal] is to [approach], which [mechanism], thereby [outcome].",
    "The convergence of [method] hinges on [property] of the objective: if [condition] holds, then [result]; otherwise, [alternative outcome].",
    "By [mechanism], [method] navigates the [property] of the landscape\u2014such as [example]\u2014thereby [benefit].",
    "A widely-adopted criterion for [decision] is [condition], which ensures [desirable property] while permitting [flexibility].",
    "We interpret [quantity] through the lens of [mathematical framework], which casts [phenomenon] as [structured form].",
    "The interplay between [notion A] and [notion B] dictates the [property] of [system], analogous to how [analogy].",
    "Under the assumption that [condition], it follows that [conclusion] \u2014 a line of reasoning that parallels the derivation of [classical result].",
    "Expanding [function] to [order] around [point] yields [approximation], from which we derive [insight] \u2014 a local analysis that nonetheless captures the essential global behavior of [system].",
    "The [objective/quantity] decomposes into [term A], which incentivizes [behavior X], and [term B], which penalizes [behavior Y] \u2014 a trade-off whose resolution determines the optimal [strategy/representation] in [domain].",
    "By [theorem/inequality], we obtain [bound/relation], which, when combined with [condition], yields [conclusion] \u2014 a chain of inequalities that traces [phenomenon] back to its fundamental [cause/property].",
    "The performance of [method] is assessed along [N] dimensions: [dimension A] as measured by [metric A], [dimension B] captured by [metric B], and [dimension C] quantified via [metric C] \u2014 a multi-faceted evaluation that reveals where [method] excels relative to [baseline] and where its limitations persist.",
    "Systematically removing [component] from the full architecture \u2014 an ablation that isolates the [function] it subserves \u2014 yields a [metric] drop of [amount], confirming that [component] contributes substantially to [capability] under [condition].",
    "The reported [metric] of [method] \u2014 [value] \u2014 is accompanied by a [confidence interval / standard deviation] of [range]; a [statistical test, e.g., paired bootstrap, McNemar] comparing [method] to [baseline] yields p = [value], [above/below] the [significance threshold] \u2014 evidence that the observed difference [is/is not] attributable to sampling variability alone.",
    "[Regularizer] introduces a [penalty] on [quantity], thereby [effect] while preserving [property].",
    "The dynamics of [quantity] under [regularization scheme] diverge from the unregularized setting in that [key difference].",
    "Absent [regularization], the [model] tends to [pathological behavior], an issue that [technique] addresses by [mechanism].",
]

print("\n=== Pattern Check ===")
for p in data["patterns"]:
    ok = p["pattern"] not in avoid_patterns
    print(f"  {'PASS' if ok else 'FAIL'}: {p['pattern'][:60]}...")
    if not ok:
        raise SystemExit(1)

# Tips avoid list
avoid_tips = [
    "用 'convex combination' 和 'affine transformation' 替代笼统的 'weighted sum'",
    "区分 'inferior to' 和 'worse than'\u2014\u2014用拉丁比较级更学术",
    "用 'smoothness' 和 'regularity' 替代 'stability' 描述训练行为",
    "用 incur 替代 cause 描述计算开销",
    "用 underpin 替代 support 表达基础性关系",
    "区分 attenuate 与 reduce 的使用场景",
    "用 'surrogate' 替代 'approximate' 描述代用优化目标",
    "用 'propagate' 替代 'pass/send' 描述信息/梯度传递",
    "用 'coincide with' 替代 'be the same as' 表达数学等价性",
    "用 'leverage' 替代 'use' 强调策略性利用",
    "用 'exhibit' 替代 'show' 描述模型的行为或性质",
    "用 'as a function of' 替代 'based on' 表达变量间的依赖关系",
    "用 'facilitate' 替代 'help' 或 'enable' 表达促进性作用",
    "用 'thereby' 连接因果链条，替代 'and thus' 或 'by doing this'",
    "用 'corroborate' 替代 'support' 或 'confirm' 表达实验证据的佐证关系",
    "用 'cascade' 替代 'chain' 或 'pass through' 描述逐层信息处理",
    "用 'alleviate' 替代 'reduce' 或 'mitigate' 表达对非量化问题的改善",
    "用 'expressive power' 而非 'capability' 描述模型的理论表示能力",
    "用 'inductive bias' 替代 'prior knowledge' 或 'assumption' 描述模型的结构性先验",
    "用 'equivariance' 替代 'invariance' 描述网络层对变换保持结构而非丢弃信息的性质",
    "用 'hierarchical composition' 替代 'layer stacking' 描述深层网络的特征层级组织特性",
    "用 'attain' 替代 'reach' 或 'achieve' 描述收敛到某点或获取某收敛率",
    "用 'traverse' 替代 'go through' 或 'move along' 描述优化路径",
    "用 'hinge on' 或 'rest upon' 替代 'depend on' 表达理论假设的依附关系",
    "用 'admit' 替代 'have' 表示矩阵或空间拥有某种分解或结构",
    "用 'characterize' 替代 'describe' 表达对数学结构或性质的精确刻画",
    "用 'non-trivial' 替代 'non-zero' 或 'meaningful' 区分平凡解与实质性结果",
    "用 'recast' 替代 'reformulate' 或 'rewrite' 表达对问题或公式的重新表述",
    "区分 'constitute' 与 'comprise' 在描述组成部分时的用法差异\u2014\u2014视角相反",
    "用 'entail' 替代 'mean' 或 'imply' 表达逻辑必然性\u2014\u2014比 imply 更强且更精确",
    "区分 'statistically significant' 与 'practically meaningful' 在实验分析中的语义差别",
    "用 'calibrate' 而非 'adjust' 或 'tune' 描述概率输出的后处理",
    "用 'attribution' 替代 'importance' 描述特征对预测的贡献分配",
    "用 'induce' 替代 'cause' 或 'create' 表达正则化引入结构性质",
    "用 'trade-off' 替代 'balance' 描述偏差-方差之间的张力关系",
    "区分 'explicit regularization' 与 'implicit regularization' 在论文写作中的语义差异",
]

print("\n=== Tips Check ===")
for t in data["tips"]:
    ok = t["tip"] not in avoid_tips
    print(f"  {'PASS' if ok else 'FAIL'}: {t['tip'][:60]}...")
    if not ok:
        raise SystemExit(1)

print("\n=== ALL CHECKS PASSED ===")
