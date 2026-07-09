# Reward Function: What Was Broken, the Fix, and the Math

## 1. Background — where the reward fits in the MDP

The paper formalizes the recommender as an MDP $(S, A, P_a, R_a)$ (Section 3, Eq. before
Eq. 1) and uses Q-learning to update a table of state-action values:

$$Q(s,a) \leftarrow Q(s,a) + \alpha\Big(r + \gamma \max_{a'} Q(s',a') - Q(s,a)\Big) \quad \text{(paper's Eq. 1)}$$

Every state $s \in \{0, \dots, 35\}$ is one of the 36 K-Means product clusters, mapped onto
a 6×6 grid. $r$ is the *only* signal telling the agent whether a transition from cluster
$s$ to cluster $s'$ was good or bad. If $r$ is always 0, the update collapses to
$Q(s,a) \leftarrow Q(s,a) + \alpha\gamma\max_{a'}Q(s',a')$ — with no external reward ever
injected, the whole table can only ever decay toward 0, never learn anything.

## 2. The original reward — item-Jaccard — and why it is structurally always zero

The original `computeMyReward(s, s')` computed:

$$R_{\text{item}}(s, s') = J\big(I(s), I(s')\big) = \frac{|I(s) \cap I(s')|}{|I(s) \cup I(s')|}$$

where $I(s)$ is the set of product IDs assigned to cluster $s$.

**Claim**: for any hard-clustering algorithm (K-Means included), $I(s) \cap I(s') =
\emptyset$ for all $s \neq s'$.

**Why**: hard K-Means assigns each data point (each product) to exactly one cluster — the
one whose centroid is closest. Formally, the clusters $\{I(0), \dots, I(35)\}$ form a
*partition* of the product catalog:

$$I(s) \cap I(s') = \emptyset \quad \forall\, s \neq s', \qquad \bigcup_{s} I(s) = \text{(full catalog)}$$

This is not a bug in the clustering — it is the definition of hard clustering. It does,
however, mean $R_{\text{item}}(s, s') = 0/|I(s) \cup I(s')| = 0$ for **every** transition
between two different clusters. The only case that gives a nonzero value is the
degenerate self-transition $s = s'$, where $J(I(s), I(s)) = 1$ — but the calling code
(`AmazonExp1.py`/`ML100KExp1.py`) explicitly sets `reward = 0` whenever `new_state ==
prevState` (a no-op/wall-bump action), so even that one nonzero case never actually
reaches the Q-update.

**Empirical confirmation** (Amazon, 36×36 = 630 cluster pairs):
```
Number of cluster PAIRS with overlapping items: 0 / 630
computeMyReward(0,1) = 0.0   computeMyReward(2,5) = 0.0   computeMyReward(10,20) = 0.0
computeMyReward(0,0) = 1.0   (the one nonzero case -- never reached by the caller)
```
This also explains why `Return` (the summed per-episode reward) measured *exactly* 0.0 for
every one of the 1,862 users evaluated across both datasets, in every run performed for
this paper, including runs that predate this revision (the leftover `#Output on
01/03/2025` comment in the original code shows `Return = 0.0` too).

## 3. The fix — Jaccard over the *users* who touched each cluster

Item sets are disjoint by construction. **User sets are not.** A product cluster's
"associated users" is the set of users who rated at least one item in that cluster:

$$U(s) = \{\, u : \exists\, i \in I(s),\ u \text{ rated } i \,\}$$

A single user typically rates many items, which typically land in *many different* item
clusters — so $U(s)$ and $U(s')$ for two different clusters routinely share members, even
though $I(s)$ and $I(s')$ never do. The new reward keeps the exact same Jaccard formula,
just applied to $U$ instead of $I$:

$$R_{\text{user}}(s, s') = J\big(U(s), U(s')\big) = \frac{|U(s) \cap U(s')|}{|U(s) \cup U(s')|}, \qquad R_{\text{user}}(s,s') \in [0, 1]$$

with the convention $R_{\text{user}}(s,s') = 0$ if $U(s) \cup U(s') = \emptyset$ (avoids
division by zero; not reached in practice since every cluster has at least one associated
user).

This preserves everything the paper already says about the reward ("reward function is
based on cluster similarity... similarity metrics such as Jaccard similarity are
employed," Section 3.1) — only the object being compared changes, from items to users.

## 4. Empirical validation before committing to the fix

Checked directly against both datasets' actual clusters before implementing:

| Dataset | Cluster pairs w/ item overlap | Cluster pairs w/ user overlap | $J(U(s),U(s'))$ mean / min / max |
|---|---|---|---|
| Amazon | 0 / 630 | 16 / 630 | 0.0009 / 0.0 / 0.167 |
| MovieLens | 0 / 630 | **630 / 630** | **0.4995** / 0.021 / 0.844 |

**MovieLens**: every single cluster pair now has a real, well-distributed reward signal —
a clean, complete fix.

**Amazon**: still mostly zero (only 16 of 630 pairs have any user overlap at all). This is
not a limitation of the *formula* — it is a direct, unavoidable consequence of Amazon's
data sparsity in this dataset (1,047 of 1,191 users, 88%, have exactly one recorded rating,
so most users only ever touch one cluster and can never contribute to an overlap with
any other cluster). No reward formula can manufacture behavioral signal that isn't present
in one-rating-per-user data. The fix is still a strict improvement over the previous
0-of-630 baseline, just a partial one, and this is reported as such rather than glossed
over.

## 5. Confirmation the fix actually changes learning dynamics

Directly inspected the Q-table after one user's training run, before vs. after the fix:

| | Before (item-Jaccard) | After (user-Jaccard, MovieLens) |
|---|---|---|
| Nonzero Q-table entries | 0 / 144 | 136 / 144 |
| Max Q-value | 0.0 | 13.91 |
| Example single-user Return | 0.0 | 9.09 |

Before the fix, `extractPolicy()`'s check `if np.max(Q[state]) > 0` was never true for any
state, so the "learned policy" always fell through to `action = env.action_space.sample()`
— i.e., a random walk, regardless of training. After the fix, the Q-table carries real
learned structure and the extracted policy is no longer guaranteed-random.

## 6. What "better" means for this change, and how it's measured

This fix targets a different failure mode than the earlier bug fixes (P0.1-P0.4, which
fixed episode-termination bookkeeping) or the `stopcount` tuning (which adjusted an
already-working mechanism). This one restores a previously-nonfunctional learning signal,
so "better" should be assessed on multiple axes, not just one number going up:

1. **Is the agent learning at all now?** — Q-table non-triviality (done, confirmed above)
   and `Return` actually being nonzero and varying across users (not a constant).
2. **Recommendation quality** — Precision, Recall, F-measure, Hit Ratio, computed the same
   way as every other result in this revision, so it's directly comparable to the
   pre-fix full-population numbers already in `Rebuttal_Draft_Sections.md`.
3. **Statistical significance** — bootstrap CIs + Mann-Whitney, same methodology as
   before, to confirm any change is real and not noise (now with the full population,
   not just a sample, so this is a strong test).
4. **Per-dataset honesty** — given the Amazon/MovieLens asymmetry established in Section
   4 above, results should be evaluated and reported *separately* per dataset rather than
   pooled, since the fix's effect size is expected to differ sharply between them.

Full before/after numbers (full population, both datasets) are in
`fullpop_eval_{Amazon,MovieLens}.csv` and summarized in `Rebuttal_Draft_Sections.md`.
