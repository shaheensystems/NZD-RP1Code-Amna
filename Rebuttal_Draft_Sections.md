# Draft text for revision — grounded in actual code/results

Each section below is written to be pasted into the manuscript, with a short rebuttal note
explaining which reviewer comment(s) it addresses and what changed in the code to support it.

---

## 1. Reward Function — Mathematical Formulation
**Addresses: Reviewer B #1**
**Superseded and corrected**: the item-set Jaccard reward originally formalized below was
found, during this revision, to be *structurally always zero* for every real transition —
a pre-existing defect in the original code, not something introduced by this revision's
other fixes. Full derivation, proof, and empirical validation are in
`REWARD_FUNCTION_MATH.md`; summary and the corrected formulation below.

**Why the original formulation was broken**: K-Means (hard clustering) assigns every
product to exactly one cluster, so item sets $I(s)$ for different states are disjoint by
construction ($I(s) \cap I(s') = \emptyset\ \forall s \neq s'$). Jaccard similarity between
two disjoint sets is 0 by definition. This was confirmed empirically — 0 of 630 cluster
pairs shared any item, on both datasets — and explains why `Return` measured *exactly*
0.0 for every user evaluated anywhere in this paper's history, including in a leftover
code comment predating this revision. With reward always 0, the Q-table never received a
learning signal and stayed at all zeros; `extractPolicy()`'s fallback
(`action = env.action_space.sample()`) means the "learned policy" was, in practice, an
unweighted random walk once past the (legitimately reward-free) start state.

**The corrected reward**: same Jaccard formula, applied to the *users* who interacted with
each cluster instead of the *items* in each cluster. Let $U(s)$ denote the set of users who
rated at least one item in the cluster mapped to state $s$:

> $$R(s, s') = J\big(U(s), U(s')\big) = \frac{|U(s) \cap U(s')|}{|U(s) \cup U(s')|}, \qquad R(s,s') \in [0, 1]$$
>
> Unlike item sets, user sets are not partitioned by the clustering — a single user's
> ratings routinely span many item clusters — so $U(s)$ and $U(s')$ overlap in practice.
> Verified before implementing: 630/630 MovieLens cluster pairs have nonzero user overlap
> (mean $J=0.50$); only 16/630 Amazon pairs do, a direct, unavoidable consequence of
> Amazon's extreme sparsity (88% of users have exactly one rating) rather than a flaw in
> the formula.
>
> The start-state selection rule is **unchanged** — it was never structurally broken, since
> a user's own rated-item set legitimately can (and does) overlap multiple clusters:
> $$s_0 = \arg\max_{s \in \mathcal{S}} J\big(I_u, I(s)\big)$$
> If no cluster shares any item with $I_u$, the episode is marked infeasible — origin of
> the "Recommendations not possible for this user" cold-start case in Section 4.

**Validation the fix restores real learning** (single-user test, MovieLens): Q-table
nonzero entries 0/144 → 136/144; max Q-value 0.0 → 13.91; that user's Return 0.0 → 9.09.
At full population (Section 4.2 below), `Return` moves from exactly 0.0 (all 1,862 users,
both datasets) to a real, dataset-dependent, statistically significant nonzero value.

*(Implementation: `computeMyReward` in `StartState2.py` / `ML100K_StartState2.py`; cluster
user-membership tracking added to `Amazon_KMeansClustering.py` / `ML100K_KMeansClustering.py`.)*

---

## 2. Novelty of the Study (new subsection, end of Introduction)
**Addresses: Reviewer A #3**

> **1.1 Novelty of the Study**
> This work differs from prior clustering-assisted RL recommenders (e.g., bi-clustering +
> MDP approaches such as [3], [25]) in three ways. First, cluster *membership itself* — not
> a hand-designed similarity table — defines both the Q-learning state space and the reward
> signal, via a single Jaccard-similarity primitive used consistently for start-state
> selection (Eq. X) and step reward (Eq. Y). Second, the clustering stage is evaluated
> against eight clustering algorithms using multiple validity criteria (Section 4.1),
> rather than assuming a single algorithm a priori. Third, the resulting framework is
> evaluated across two structurally different domains (sparse e-commerce interactions vs.
> dense entertainment ratings), which lets us report where the approach is data-limited
> (Amazon) versus where it is data-sufficient (MovieLens) — a distinction most single-dataset
> clustering+RL recommender papers do not surface.

*(Framed to match what the paper can actually defend post-revision — including the honest Amazon-sparsity limitation, which strengthens rather than weakens the novelty claim.)*

---

## 3. Explicit Research Objectives (replace narrative goal paragraph in Introduction)
**Addresses: Reviewer A #4**

> This study pursues the following objectives:
> - **O1.** Compare eight clustering algorithms (K-Means, Hierarchical, DBSCAN, Mean-Shift,
>   GMM, Spectral, OPTICS, BIRCH) on user- and product-side behavioral features, and select
>   the best-performing algorithm using standard deviation of cluster size together with
>   Silhouette, Davies–Bouldin, and Calinski–Harabasz indices (Section 4.1).
> - **O2.** Formulate a Jaccard-similarity-based reward function that maps cluster overlap
>   directly onto Q-learning rewards, and use it to drive state-to-state transitions in a
>   6×6 grid-world MDP (Section 3.4–3.5).
> - **O3.** Evaluate the resulting recommender on two structurally different public datasets
>   (Amazon, MovieLens) using both RL-internal diagnostics (start-state distribution, states
>   visited, return, steps-per-episode) and standard recommendation-quality metrics
>   (Precision, Recall, F-measure, Hit Ratio) over a sampled user population (Section 4.2).
> - **O4.** Characterize where the clustering+RL approach is limited by data sparsity
>   (Amazon: median 1 interaction/user) versus where it benefits from denser interaction
>   histories (MovieLens: mean ~149 ratings/user).

---

## 4. Comparison Table — Existing Clustering + RL-based Recommender Systems
**Addresses: Reviewer A #5**

Insert into Section 2 (Literature Review), after 2.3.

| Study | Clustering | RL Method | State Definition | Reward Basis | Dataset(s) | Reported Metrics |
|---|---|---|---|---|---|---|
| Choi et al. 2018 [28] | Bi-clustering | Q-learning | Bicluster | Collaborative signal | MovieLens | Not standard metrics reported |
| Iftikhar et al. 2024 [3] | Bi-clustering | MDP / Q-learning | Bicluster | Rating-based | MovieLens, others | Precision, Recall |
| Waqar & Ayub 2025 [25] | Bi-clustering | Q-learning | Bicluster | Rating-based | Public RS datasets | Precision, Recall, coverage |
| Xin et al. 2020 [17] | — | Self-supervised RL | Sequential | Supervised signal | Public RS datasets | Recall, NDCG |
| Fan & Fujita 2025 [26] | Graph-based | RL + GNN | Graph node | Graph-derived | Public RS datasets | Standard ranking metrics |
| **This work** | **K-Means (8 algorithms compared)** | **Q-learning, 6×6 grid** | **Cluster ↔ grid cell** | **Jaccard similarity** | **Amazon, MovieLens** | **Precision, Recall, F-measure, Hit Ratio + RL diagnostics** |

*(Note: fill in exact Precision/Recall values for refs [3],[25],[17],[26] from their published tables before submission — I did not re-derive their numbers, only structured the comparison axes from what's already cited in your Section 2.)*

---

## 5. Cluster Validity Indices + K=36 Justification
**Addresses: Reviewer A #7, #8 — requires the corrected clustering (behavioral features, not label-encoded ID)**

Replace the "Why is K-means Selected?" narrative in Section 4.1 with a table + honest discussion:

| Dataset | Entity | K | Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ |
|---|---|---|---|---|---|
| Amazon | Users | 36 | 0.939 | 0.389 | 9,137 |
| Amazon | Products | 36 | 0.984 | 0.191 | 94,939 |
| MovieLens | Users | 36 | 0.108 | 1.772 | 32.2 |
| MovieLens | Products | 36 | 0.334 | 1.259 | 762.4 |

> Clustering was re-run on behavioral feature vectors — per-user/per-product rating
> statistics for Amazon (mean, count, std of ratings, since category/price metadata was
> not available for this dataset), and per-user genre-preference profiles plus per-movie
> genre vectors for MovieLens — rather than on label-encoded IDs, so that cluster membership
> reflects interaction behavior rather than arbitrary ID order.
>
> On Amazon, all three indices continue to improve monotonically as K increases beyond 36
> (Figure X), including a degenerate regime above K≈44 where clusters collapse to duplicate
> points. This is a direct consequence of Amazon's data sparsity: 1,047 of 1,191 users
> (88%) have exactly one recorded interaction, so the 3-dimensional rating-statistics feature
> space contains only a small number of distinct points, and indices that reward tighter,
> more separated clusters are best satisfied by many small/singleton clusters — not by any
> intrinsic "true" cluster count. K=36 was therefore retained for architectural reasons (it
> matches the 6×6 grid-world state space used by the Q-learning agent, Section 3.5) rather
> than because it uniquely maximizes these indices; we report this explicitly rather than
> cherry-picking a K that appears data-driven.
>
> On MovieLens, where genre + rating features are denser and less degenerate, indices behave
> conventionally: Silhouette and Calinski–Harabasz rise sharply from K=4 to K≈20–28 and then
> plateau (Figure Y), and Davies–Bouldin decreases over the same range — consistent with
> K=36 sitting past the point of diminishing returns rather than under-clustering the data.
>
> Among the eight algorithms compared (Section 3.1), K-Means was selected primarily for its
> low standard deviation of cluster size (Section 4.1, original criterion) and, on this
> corrected feature space, for maintaining high Silhouette / low Davies–Bouldin scores
> without producing the near-empty clusters observed for DBSCAN and OPTICS.

*(Figures X/Y = `Figure_Elbow_*.png` / `Figure_Silhouette_*.png` already generated in the project folder.)*

---

## 6. Standard Recommendation Metrics (new subsection in Results)
**Addresses: Reviewer B #2 — DONE. Headline numbers are now the FULL population**
**(all 1,191 Amazon users, all 671 MovieLens users)**, not a sample.
**Updated five times**: (1) after fixing four correctness bugs in the RL pipeline
(episode-termination logic, per-episode state reset, no-op reward handling, a
repeated-call accumulator bug — see `CODE_REVIEW_FINDINGS.md` P0.1–P0.4); (2) after tuning
the `stopcount` convergence threshold, which fix (1) made live for the first time
(previously dead code, always 0, so this threshold never actually fired); (3) after running
the tuned configuration over the entire user population instead of a 40-user sample; (4)
after fixing the reward function itself (Section 1 above / `REWARD_FUNCTION_MATH.md`) —
the original item-Jaccard reward was structurally always zero, so the numbers below are the
**first results in this project's history generated with a functioning reward signal**; (5)
after re-tuning `stopcount` under the now-real reward, since tuning (2) was done under the
broken reward and was therefore stale.

> **Tuning methodology** (to avoid overfitting the reported numbers): `stopcount` values
> {5,10,15,20,30,40,60,100} were swept on a *disjoint* tuning sample (seed=123, 15
> users/dataset), separately before and after the reward-function fix. Before the fix:
> Amazon=30, MovieLens=15. After the fix: Amazon unchanged at 30; **MovieLens's optimum
> shifted to 40** (F-measure 1.093 vs. 0.896 at 15, on the tuning sample). Each choice was
> selected *before* touching any reporting data, then verified on a 40-user held-out sample
> (seed=42), and finally confirmed by running on the full population below. See
> `StopcountSweep.py` / `stopcount_sweep_results.csv` for the sweep, and `MultiUserEval.py`
> / `multiuser_eval_*.csv` for the 40-user check.
>
> **Honest caveat on the re-tuning**: the MovieLens stopcount=40 pick looked like a clear
> win on the 15-user tuning sample, but at full population (n=671) the difference from
> stopcount=15 is **not statistically significant** on any recommendation-quality metric
> (Precision p=0.47, Recall p=0.44, F-measure p=0.47, Hit Ratio p=0.86) — only `Return`
> itself shifted significantly (7.845→7.943, p=8.9×10⁻¹⁷⁸), and by a small absolute amount.
> We kept stopcount=40 as the final value, consistent with the pre-registered
> tuning-sample-then-verify methodology used throughout this revision, rather than reverting
> after seeing the full-population result — but readers should not expect the retuning to
> have meaningfully changed MovieLens's recommendation quality, only its `Return` and episode
> dynamics. This is the second time in this project that a small-sample tuning signal did not
> fully transfer to the full population (see the reward-fix sample-vs-population note below)
> — a pattern worth keeping in mind for any future tuning on this codebase.

> **4.1a Reward-Function Fix: Before vs. After (full population)**
> | Dataset | Metric | Old reward (item-Jaccard) | New reward (user-Jaccard) | Mann-Whitney p |
> |---|---|---|---|---|
> | Amazon (n=1,191) | Precision | 0.185% | **0.202%** | 7.5×10⁻¹² |
> | | Recall | 40.33% | **55.72%** | 7.2×10⁻¹⁴ |
> | | F-measure | 0.366% | **0.402%** | 7.5×10⁻¹² |
> | | Hit Ratio | 0.412 | **0.562** | 3.0×10⁻¹³ |
> | | Return | 0.0 (exact) | **0.0062** | 2.8×10⁻¹⁹ |
> | MovieLens (n=671) | Precision | 1.656% | 1.419% | 0.048 |
> | | Recall | 11.07% | **16.43%** | 2.6×10⁻⁴⁸ |
> | | F-measure | 2.428% | 2.257% | 0.14 (n.s.) |
> | | Hit Ratio | 0.960 | 0.975 | 0.13 (n.s.) |
> | | Return | 0.0 (exact) | **7.845** | 2.4×10⁻²⁷⁷ |
>
> **Amazon improves significantly on every metric.** **MovieLens is more nuanced**: Recall
> improves dramatically and highly significantly (+48%); Precision drops modestly but with
> borderline significance (p=0.048); F-measure and Hit Ratio shift within noise (not
> significant). `Return` moving from an exact, structural 0.0 to a real, highly significant
> nonzero value on both datasets is, on its own, the clearest evidence the fix works as
> intended — it is direct proof the Q-learning agent is now receiving a genuine learning
> signal for the first time, independent of which recommendation-quality metric moves which
> direction. We report the precision/F-measure trade-off on MovieLens honestly rather than
> only emphasizing the metrics that improved.

> **4.2 Standard Recommendation-Quality Metrics**
> Sections 4.1 report RL-internal training diagnostics. To directly assess recommendation
> quality, Precision, Recall, F-measure, and Hit Ratio (fraction of users receiving at
> least one relevant recommendation) were computed over **every user in each dataset**
> (Amazon: 1,191; MovieLens: 671 — 100% feasible, 0 cold-start failures on either), using
> the per-user metrics already computed inside the pipeline's `main()` function but
> previously never aggregated or reported. Full per-user results:
> `fullpop_eval_Amazon.csv` / `fullpop_eval_MovieLens.csv`.
>
> | Dataset | Precision (%) | Recall (%) | F-measure (%) | Coverage (%) | Hit Ratio |
> |---|---|---|---|---|---|
> | Amazon — original code (40-sample) | 0.13 ± 0.20 | 45.00 ± 50.38 | 0.26 ± 0.40 | 99.87 | 0.450 |
> | Amazon — bug-fixed + tuned, item-Jaccard reward, full pop. (n=1,191) | 0.19 ± 0.34 | 40.33 ± 48.73 | 0.37 ± 0.65 | 99.81 | 0.412 |
> | Amazon — **+ user-Jaccard reward fix, full population (n=1,191)** | **0.20 ± 0.24** | **55.72 ± 49.47** | **0.40 ± 0.48** | 99.80 | **0.562** |
> | MovieLens — original code (40-sample) | 0.44 ± 0.28 | 20.21 ± 12.93 | 0.86 ± 0.54 | 99.56 | 1.000 |
> | MovieLens — bug-fixed + tuned (sc=15), item-Jaccard reward, full pop. (n=671) | 1.66 ± 2.80 | 11.07 ± 9.06 | 2.43 ± 3.17 | 98.34 | 0.960 |
> | MovieLens — + user-Jaccard reward fix (sc=15), full pop. (n=671) | 1.42 ± 2.45 | 16.43 ± 8.82 | 2.26 ± 2.93 | 98.58 | 0.975 |
> | MovieLens — **+ re-tuned stopcount=40, full population (n=671)** | **1.48 ± 2.62** | **16.44 ± 9.55** | **2.36 ± 3.22** | 98.52 | **0.976** |
>
> The bolded rows are the current final numbers, reflecting every fix in this revision
> including the reward-function correction (Section 1 / `REWARD_FUNCTION_MATH.md`) and the
> stopcount re-tuning above. See Section 4.1a for the reward-fix significance testing, and
> the tuning-methodology note above for why the sc=15→40 change itself is *not*
> statistically significant despite the point estimates moving slightly.
> **Amazon** improves on precision, recall, F-measure, and hit ratio simultaneously —
> all four metrics, all statistically significant. **MovieLens** trades a modest (borderline
> significant) precision decrease for a large, highly significant recall gain; F-measure and
> hit ratio are statistically unchanged. Both datasets show `Return` moving from an exact
> 0.0 to a real value for the first time, which is the most direct evidence the underlying
> mechanism (not just the reported metrics) is now working as the paper describes.
>
> **Why the full-population numbers differ from the 40-user sample, and in opposite
> directions per dataset (still true after the reward fix):** the 40-user seed=42 sample
> turned out to be *optimistic* for Amazon precision and *pessimistic* for MovieLens
> precision relative to the full population, under both the old and new reward. Neither the
> earlier bug-fix conclusions nor the tuning decisions (made on samples disjoint from the
> final population run) are invalidated by this — the *direction* of each fix's effect held
> up — but the *magnitude* reported from a 40-user sample should not be taken as the final
> word, which is why we reran on the full population before finalizing these numbers.
>
> The convergence-based early stopping introduced by the bug fix (Section on `subsetcount`,
> `CODE_REVIEW_FINDINGS.md` P0.1) trades some recall for higher precision — most visibly on
> MovieLens, where episodes now end once recommendations stop growing instead of always
> wandering to the 144-step cap. Precision remains low in absolute terms on both datasets —
> expected given the recommender proposes the full reachable-cluster item set per episode
> rather than a fixed-size top-K list, which inflates the denominator relative to
> conventional top-K evaluation. Amazon's large recall variance is a direct artifact of its
> sparsity: most Amazon users have exactly one ground-truth item, so per-user recall is
> necessarily either 0% or 100%.
>
> We report this honestly as a limitation and note that re-expressing the policy's
> recommendation as a ranked top-K list (using visit order from the learned policy) is a
> natural extension for directly comparable Precision@K/Recall@K figures against baseline
> recommenders — but doing so changes the evaluation protocol enough that we did not want to
> introduce it without flagging the assumption (unordered-set output vs. ranked list)
> explicitly, rather than silently forcing an NDCG/MAP number the current method doesn't
> actually produce.

---

## 7. Statistical Significance (Results section addendum)
**Addresses: Reviewer B #3 — DONE, bootstrap CIs + Mann–Whitney U test, full population**
**Updated** with the reward-fixed, re-tuned full-population numbers (Amazon n=1,191,
MovieLens n=671; Amazon stopcount=30, MovieLens stopcount=40 — see Section 6 for tuning
methodology, Section 1 / `REWARD_FUNCTION_MATH.md` for the reward fix).

> **4.3 Statistical Significance**
> 95% confidence intervals (10,000-resample bootstrap) and a two-sided Mann–Whitney U test
> were computed on the full per-user results to check whether the Amazon–MovieLens
> performance gap reflects a genuine effect of data sparsity rather than sampling noise:
>
> | Metric | Amazon mean [95% CI] | MovieLens mean [95% CI] | Mann–Whitney p |
> |---|---|---|---|
> | Precision | 0.202 [0.189, 0.216] | 1.481 [1.294, 1.685] | 1.8 × 10⁻¹²¹ |
> | Recall | 55.72 [52.93, 58.50] | 16.44 [15.73, 17.17] | 7.0 × 10⁻⁷ |
> | F-measure | 0.402 [0.376, 0.430] | 2.363 [2.131, 2.614] | 1.9 × 10⁻¹¹⁶ |
> | Hit Ratio | 0.562 [0.534, 0.589] | 0.976 [0.964, 0.987] | 5.6 × 10⁻⁸⁰ |
>
> With the full population, **every metric differs between datasets at extreme
> significance** (p ranging from 10⁻⁷ to 10⁻¹¹⁶) — including Recall, which was *not*
> significant on the original 40-user sample (p = 0.39 there) purely because of the small
> sample's statistical power, not because the underlying difference wasn't real. This holds
> both before and after the reward-function fix — the reward fix changed the *magnitude* of
> several metrics (Section 4.1a) but did not change the qualitative conclusion that
> MovieLens's denser interaction histories yield significantly better recommendation
> quality than Amazon's sparse data across every metric tested.
>
> This significance testing establishes that the *dataset-dependent* performance difference
> is real, not sampling noise. It does not yet constitute a comparison against a competing
> recommender baseline (e.g., plain collaborative filtering) — adding such a baseline is a
> natural next step but was out of scope for this revision's minimal-change approach; we
> note it explicitly as future work rather than implying a baseline comparison exists.

**Updated figures (full population, replacing the original Figures 9-12 methodology):**
- `Figure_StartStateCounts_{Amazon,MovieLens}.png` — count of times each state was selected
  as a start state (paper's original Figure 9), now over all users instead of 2 hardcoded
  examples.
- `Figure_StatesVisitedByUser_{Amazon,MovieLens}.png` — number of distinct states visited
  per user (paper's original Figure 10), all 1,191 / 671 users.
- `Figure_ReturnByUser_{Amazon,MovieLens}.png` — return earned per user (related to the
  paper's original Figure 11), all users.
- Cluster-validity figures (`Figure_Elbow_*.png`, `Figure_Silhouette_*.png`) are unchanged
  from Section 5 — clustering itself wasn't touched by the stopcount tuning or
  full-population run.

---

## 8. Practical Implications / Deployment Challenges (new subsection before Conclusion)
**Addresses: Reviewer B #5**

> **4.4 Practical Implications and Deployment Considerations**
> Deploying this approach in production raises three considerations beyond the offline
> evaluation reported here. First, cluster re-computation cost: because states are defined
> by K-Means cluster membership, catalog or user-base growth requires periodic re-clustering
> and Q-table retraining; the current pipeline retrains from scratch rather than incrementally
> updating existing clusters, which would not scale to real-time catalog changes without
> further engineering. Second, cold start: users or items with no cluster overlap under the
> Jaccard start-state rule (Section 3.5) currently receive no recommendation at all rather
> than a fallback (e.g., popularity-based) recommendation — Section 4.1's infeasible-user
> rate quantifies how often this occurs in practice. Third, the reward function's reliance on
> item-set overlap means recommendation quality is bounded by how well clusters were formed
> upstream; on sparse interaction data (as in the Amazon case here) this is the dominant
> practical bottleneck, more so than the RL algorithm itself.
