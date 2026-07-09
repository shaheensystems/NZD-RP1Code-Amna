# Draft text for revision — grounded in actual code/results

Each section below is written to be pasted into the manuscript, with a short rebuttal note
explaining which reviewer comment(s) it addresses and what changed in the code to support it.

---

## 1. Reward Function — Mathematical Formulation
**Addresses: Reviewer B #1**

Insert into Section 3.5 (Integration of Reinforcement Learning), replacing the qualitative
description of the Jaccard-based reward.

> Let $I(s)$ denote the set of product IDs contained in the cluster mapped to state $s$.
> The immediate reward for a transition from state $s$ to state $s'$ is the Jaccard
> similarity between their item sets:
>
> $$R(s, s') = J\big(I(s), I(s')\big) = \frac{|I(s) \cap I(s')|}{|I(s) \cup I(s')|}, \qquad R(s,s') \in [0, 1]$$
>
> The start state for a target user $u$ with rated-item set $I_u$ is likewise chosen by
> maximizing Jaccard similarity between $I_u$ and every candidate cluster's item set:
>
> $$s_0 = \arg\max_{s \in \mathcal{S}} J\big(I_u, I(s)\big)$$
>
> If no cluster shares any item with $I_u$ (i.e. $J(I_u, I(s)) = 0$ for all $s$), the episode
> is marked infeasible and no recommendation is issued for that user — this is the origin of
> the "Recommendations not possible for this user" cold-start case reported in Section 4.
> This reward formulation rewards the agent for moving toward clusters whose product
> composition overlaps with clusters already visited (and, at the start, with the user's own
> history), operationalizing "cluster similarity" from Section 3.1 as a concrete, bounded
> quantity that plugs directly into the Q-learning update in Eq. (1).

*(This matches `computeMyReward` and `computeJaccard` in `StartState2.py` / `ML100K_StartState2.py` exactly — no new logic was introduced, only formalized.)*

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
**Addresses: Reviewer B #2 — DONE, real results from 40 sampled users/dataset**
**Updated twice**: first after fixing four correctness bugs in the RL pipeline
(episode-termination logic, per-episode state reset, no-op reward handling, a
repeated-call accumulator bug — see `CODE_REVIEW_FINDINGS.md` P0.1–P0.4); then again
after tuning the `stopcount` convergence threshold, which the first fix made live for the
first time (previously dead code, always 0, so this threshold never actually fired). Full
tuning methodology and honest before/after numbers below — including where tuning did
**not** produce a win, reported as such rather than hidden.
>
> **Tuning methodology** (to avoid overfitting the reported numbers): `stopcount` values
> {5,10,15,20,30,40,60,100} were swept on a *disjoint* tuning sample (seed=123, 15
> users/dataset, distinct from the seed=42 sample used for all reported numbers). The
> best-performing value per dataset (by F-measure) was selected *before* touching the
> seed=42 sample, then verified once on it — Amazon=30, MovieLens=15. See
> `StopcountSweep.py` and `stopcount_sweep_results.csv` for the full sweep.

> **4.2 Standard Recommendation-Quality Metrics**
> Sections 4.1 report RL-internal training diagnostics. To directly assess recommendation
> quality, Precision, Recall, F-measure, item Coverage, and Hit Ratio (fraction of users
> receiving at least one relevant recommendation) were computed over 40 real, randomly
> sampled users per dataset (seed=42, fully reproducible run-to-run), using the per-user
> metrics already computed inside the pipeline's `main()` function but previously never
> aggregated or reported.
>
> | Dataset | Precision (%) | Recall (%) | F-measure (%) | Coverage (%) | Hit Ratio |
> |---|---|---|---|---|---|
> | Amazon (original code) | 0.13 ± 0.20 | 45.00 ± 50.38 | 0.26 ± 0.40 | 99.87 | 0.450 |
> | Amazon (bug-fixed, stopcount=30) | **0.29 ± 0.41** | **52.08 ± 49.38** | **0.57 ± 0.80** | 99.71 | **0.550** |
> | MovieLens (original code) | 0.44 ± 0.28 | 20.21 ± 12.93 | 0.86 ± 0.54 | 99.56 | 1.000 |
> | MovieLens (bug-fixed, stopcount=15) | 0.44 ± 0.28 | 10.13 ± 7.32 | 0.83 ± 0.54 | 99.56 | 0.975 |
>
> **Amazon improves on all four metrics** after the bug fixes and tuning — Precision +117%,
> Recall +16%, F-measure +115%, Hit Ratio +22%, all verified on the same held-out seed=42
> sample the tuning phase never saw. **MovieLens is materially unchanged on Precision and
> F-measure** (within noise of the original numbers) **but Recall and Hit Ratio are lower**
> (20.2%→10.1% and 100%→97.5%) — the convergence-based early stopping that helps Amazon cuts
> MovieLens episodes short before the agent accumulates as many relevant items. We tested
> stopcount up to 100 for MovieLens (`stopcount_sweep_results.csv`) and recall never
> recovered to the original level, so this is reported as a genuine trade-off introduced by
> fixing the termination-condition bug, not a tuning shortfall.
>
> Precision is low in absolute terms on both datasets — expected given the recommender
> proposes the full reachable-cluster item set per episode rather than a fixed-size top-K
> list, which inflates the denominator relative to conventional top-K evaluation. Amazon's
> large recall variance is a direct artifact of its sparsity: most sampled Amazon users have
> exactly one ground-truth item, so per-user recall is necessarily either 0% or 100%.
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
**Addresses: Reviewer B #3 — DONE, bootstrap CIs + Mann–Whitney U test, 40 users/dataset**
**Updated** with the final bug-fixed + tuned numbers (Amazon stopcount=30, MovieLens
stopcount=15; see Section 6 for the full before/after and tuning methodology).

> **4.3 Statistical Significance**
> 95% confidence intervals (10,000-resample bootstrap) and a two-sided Mann–Whitney U test
> were computed to check whether the Amazon–MovieLens performance gap reflects a genuine
> effect of data sparsity rather than sampling noise:
>
> | Metric | Amazon mean [95% CI] | MovieLens mean [95% CI] | Mann–Whitney p |
> |---|---|---|---|
> | Precision | 0.287 [0.169, 0.417] | 0.435 [0.354, 0.524] | 5.2 × 10⁻⁴ |
> | Recall | 52.08 [37.08, 67.08] | 10.13 [8.04, 12.51] | 0.393 (n.s.) |
> | F-measure | 0.567 [0.335, 0.824] | 0.831 [0.677, 1.000] | 1.9 × 10⁻³ |
> | Hit Ratio | 0.550 [0.400, 0.700] | 0.975 [0.925, 1.000] | 9.3 × 10⁻⁶ |
>
> Precision, F-measure, and Hit Ratio still differ between datasets at p < 0.01 after the
> bug fixes and tuning — the denser MovieLens interaction histories continue to yield
> significantly better recommendation quality than the sparse Amazon data. Recall does not
> differ significantly (p = 0.393): Amazon's recall variance widened further under
> stopcount=30 (37–67% CI) since it now recommends larger, more successful item sets for
> the sparse-data users who do get a hit, overlapping MovieLens's narrower 8–13% band. This
> non-significant result should not be read as "sparsity doesn't matter" — Precision,
> F-measure, and Hit Ratio all disagree — but as recall specifically being a noisy,
> near-binary quantity on Amazon's 1-item-per-user sparsity regime regardless of tuning.
>
> This significance testing establishes that the *dataset-dependent* performance difference
> is real, not sampling noise. It does not yet constitute a comparison against a competing
> recommender baseline (e.g., plain collaborative filtering) — adding such a baseline is a
> natural next step but was out of scope for this revision's minimal-change approach; we
> note it explicitly as future work rather than implying a baseline comparison exists.

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
