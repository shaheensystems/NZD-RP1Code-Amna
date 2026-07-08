# Code Review Findings & Improvement Plan

Scope: full `NZD-RP1Code-Amna` codebase, including files not touched in the reviewer-response
pass (Rebuttal_Draft_Sections.md work). Findings are grouped by severity/type, each with
file:line and why it matters. A prioritized plan follows at the end.

---

## A. Likely correctness bugs (not just style)

### A1. `env.subsetcount` goal-tracking is dead — actual stopping condition is a different, cruder mechanism
`AmazonExp1.py:239-262` / `ML100KExp1.py` (same pattern), and mirrored in `extractPolicy` /
`applyPolicy`.

`computeGoal()` (in `StartState2.py`) is designed to increment a `subsetcount` when a newly
visited cluster contributes no new items — the intended convergence signal ("if
subsetcount>=TH then goal=True", per the comment at the top of `StartState2.py`). But the
call site does:
```python
env.Recitems, sscount = StartState2.computeGoal(itemset, env.GridPos[row][col], env.Dictobj, env.subsetcount)
```
`sscount` is **never written back** to `env.subsetcount` (in `extractPolicy` the line
`#env.subsetcount = sscount` is even explicitly commented out — `AmazonExp1.py:369`). So
`env.subsetcount` stays 0 for the entire episode, and the actual episode-termination
condition ends up being an unrelated heuristic: `statecount == env.stopcount` (agent
revisits the same grid cell 10 times) or a hard step cap (`step == 144`). `env.Recitems`
(the recommended-items accumulator) is still being built correctly via `computeGoal`, but
the mechanism that was supposed to detect "recommendations have converged" is vestigial.
**Practically**: episodes always run until the 10-revisit or 144-step cap, not until any
notion of recommendation convergence — worth confirming this matches the authors' intent,
since the code and the design comment disagree.

### A2. `learnPolicy`'s `strt` counter is scoped across all 100 episodes, not reset per episode
`AmazonExp1.py:176` (`strt=0` set once, outside `for i_episode in range(num_episodes)`),
incremented at `AmazonExp1.py:262`.

`if (strt == 0):` is meant to special-case "first step, start with an empty recommendation
set" (`itemset = set()`), but because `strt` is a function-level (not per-episode) counter,
this branch only fires once — on the very first step of episode 0. From episode 1 onward,
`env.Recitems` carries over from wherever the previous episode left off, rather than
starting fresh each episode. This looks unintentional: each episode should plausibly start
recommendation-tracking from scratch. Worth a `strt = 0` reset at the top of the
`for i_episode` loop, or replacing the `strt==0` check with `step==1`.

### A3. `env.step()`'s returned `reward` is effectively discarded, then partially reused on a no-op step
`AmazonExp1.py:86-123` (env.step), `228-259` (caller).

`env.step()` computes and returns a `reward` (always 0, or 1 only if `done` was already true
*before* this step). The caller immediately recomputes its own `reward` via
`StartState2.computeMyReward(...)` — but **only when `new_state != prevState`**
(`AmazonExp1.py:253`). When the agent bounces off a grid boundary (`new_state == prevState`,
e.g. pressing UP from row 0), the code falls through to whatever stale `reward` value is
left over from `env.step()`'s own computation *or, worse, from the previous loop
iteration* (Python doesn't reset `reward` between iterations of the `while True:` loop) —
`reward` is a leftover local variable at that point, not intentionally 0. That value feeds
directly into the Q-learning update (`AmazonExp1.py:264-265`). This is a real (if
low-impact, since it only affects no-op boundary actions) reward-leakage bug: a stale
reward from a prior step can be applied to the Q-update after a no-op action.

### A4. Module-level global accumulator lists never reset between calls to `run()`
`AmazonExp1.py:542` / `ML100KExp1.py:542` — `precisioni=[]; recalli=[]; Fmeasurei=[]; coveragei=[]; Return=[]`
declared once at import time, appended to inside `run()` (`AmazonExp1.py:553-557`), then
averaged over `range(0, len(users))` (`AmazonExp1.py:564`).

If `run()` is ever called twice in the same process (e.g. from a notebook, or from
`Amazon_RLRecommender1.py` after some other code already imported and used `AmazonExp1`),
these lists keep growing across calls, but the averaging loop only reads indices
`0..len(users)-1` — i.e. **the stale results from the first call**, not the new call's
results. Any second invocation silently reports first-run numbers. Low risk today (nothing
currently calls `run()` twice), but a real trap for anyone extending this code.

### A5. `getRating()` in both `*_RLRecommender1.py` files is dead and, for MovieLens, wrong
`ML100K_RLRecommender1.py:44-50`: compares `data[x][0]==usrid` and `data[x][1]==itmid`, but
for the MovieLens column order (`movieId, title, year, genres, userId, rating`), column 0 is
`movieId` and column 1 is `title` — the parameter names (`usrid`, `itmid`) don't match what's
actually being compared. Currently unreachable (function is never called — confirmed via
repo-wide grep), so harmless today, but a landmine if someone wires it up later believing it
works.

---

## B. Dead code / unused code (should be deleted, not left in place)

| Item | Location | Evidence |
|---|---|---|
| `Triangle.py` entire module (278 lines: `triangle`, `triangle2`, `triangle3`, `computePCC`, `urp`) | `Triangle.py` | Only referenced by `computeSim()`, which is itself never called from the active pipeline |
| `computeSim()` | `StartState2.py:15-26`, `ML100K_StartState2.py` | Never called outside its own file |
| `computeGoal2()`, `computeGoal3()` | `StartState2.py:107-178`, `ML100K_StartState2.py` | Only `computeGoal()` is actually called from `AmazonExp1.py`/`ML100KExp1.py` |
| `computeCoverage()` (non-`2`) | `AmazonExp1.py:440-456`, `ML100KExp1.py` | Only `computeCoverage2` is called; the call site even has it commented out (`AmazonExp1.py:509`) |
| `actions = {...}` dict | `AmazonExp1.py:431-437` | Unused, and its Up/Left/Down/Right→0/1/2/3 mapping **contradicts** the `TwoDGridWorld` class constants (UP=0, DOWN=1, LEFT=2, RIGHT=3) — if anyone starts using this dict, it'll silently swap actions |
| `ml100k_list`, `FT_list`, `list1..list4`, `Ilist`, `uniq` (~57KB of pasted numeric literals + scratch computations) | `Listcount.py:52-99` | Executes at **import time** on every one of the 5 modules that `import Listcount`; not used by any live code path (only referenced in commented-out lines) |
| `rat2, frat, fitm, itm2, fusr, chkusr` | `Amazon_RLRecommender1.py:11`, `ML100K_RLRecommender1.py:9` | Declared, never referenced again |
| `getRating()` | both `*_RLRecommender1.py` | Never called (see A5) |
| `ListPlot.py` (entire file) | `ListPlot.py` | Standalone script hardcoding two ~26,000-character result arrays (`list1`, `list2`) with no function wrapper; not imported anywhere — a one-off plotting script that happens to live in the package |

---

## C. Import-time side effects (already partially fixed, some remain)

Already fixed in the reviewer-response pass: `ML100K_KMeansClustering.py`'s bottom demo
block, dead filepaths in `AmazonExp1.py`/`ML100KExp1.py`/the two `*_KMeansClustering.py`
files.

**Still broken** (unfixed, would crash on `import` today):
- `Read_Amazon.py:14-16` — `read_Dataset()` is called at module level (line 47) against a
  hardcoded dead path `F:\Thesis Supervised\...`; `from Read_Amazon import *` in
  `Amazon_RLRecommender1.py:6` means importing that file crashes immediately.
- `Amazon_RLRecommender1.py` / `ML100K_RLRecommender1.py` — entire files are top-level
  scripts (CSV loads, clustering, full-dataset RL loop, blocking `plt.show()` calls) with no
  `if __name__ == '__main__':` guard and dead `F:\...` paths (lines 13, 57 / 12, 53).
  Importing either file runs a full experiment immediately and hangs on the final
  `plt.show()` if there's no display.

These are actually the scripts that produced the paper's original Figures 9–12 (they loop
over **every** user in the dataset, not a sample) — more complete than the
`MultiUserEval.py` sampling harness added during the reviewer-response pass, but currently
non-functional. Worth reconciling: either restore these as the canonical full-population
evaluation (after fixing the path/guard issues) and treat `MultiUserEval.py` as a fast
sanity-check subset, or retire them in favor of the sampled harness — right now both exist
and disagree about which is authoritative.

---

## D. Design / maintainability

- **`learnPolicy` / `extractPolicy` / `applyPolicy` are ~80% duplicated logic**, each
  reimplementing the step/reward/goal-tracking loop with small, easy-to-miss divergences
  (e.g. `extractPolicy` appends `state` to `states_visited` before advancing, `learnPolicy`
  appends `new_state`). Any bug fix has to be manually replicated across 3 functions × 2
  dataset variants = 6 places. A shared `_run_episode(env, Q_or_policy, mode)` helper would
  remove this.
- **`AmazonExp1.py` and `ML100KExp1.py` are ~599-line near-duplicates** differing mainly in
  which `*_KMeansClustering` / `*_StartState2` module they import. Same for
  `Amazon_RLRecommender1.py` / `ML100K_RLRecommender1.py`, and `StartState2.py` /
  `ML100K_StartState2.py` (which, per a `diff`, differ only in `Cluster_Items` vs.
  `Cluster_UniqItems` dict keys — functionally identical since both are immediately wrapped
  in `set(...)`). A single parameterized module (dataset config passed in) would halve the
  code surface and eliminate drift risk.
- **Magic numbers**: grid `stopcount=10`, step cap `144`, `TH=-1` in `computeJaccard`,
  `subsetcount` threshold — none are named constants or explained inline.
- **`torch` is used only for `torch.zeros`/`torch.rand`/`torch.max`/`torch.arange`** on a
  36×4 Q-table — i.e. plain tabular Q-learning with no autodiff/GPU use. This pulls in a
  ~200MB+ dependency for what NumPy (already a dependency) would do natively, and was the
  single largest, slowest, most fragile part of setting up the environment for this review
  (hit a OneDrive file-lock failure on first install attempt).
- **`computeCoverage2`'s "coverage" is really a novelty/non-overlap metric**
  (`AmazonExp1.py:458-475`: fraction of *recommended* items **not** already in the user's
  rated history), not "item coverage" in the standard recommender-systems sense (% of
  catalog reachable across all users). The paper reports this as "coverage ≈ 99%" — worth
  renaming/clarifying before a reviewer reads it as catalog coverage and is confused why
  it's so uniformly high regardless of dataset sparsity.

## E. Reproducibility / environment

- **No `requirements.txt` / `pyproject.toml` anywhere in the repo.** Every dependency
  (pandas, scikit-learn, numpy, matplotlib, gymnasium, torch, openpyxl, scipy) had to be
  inferred from `import` statements and installed manually. A pinned `requirements.txt`
  would make this reproducible for a reviewer or co-author on a fresh machine.
- **No random seed on the RL side.** `KMeans(..., random_state=42)` is seeded, but
  `torch.rand(...)` (ε-greedy exploration) and `env.action_space.sample()` (gymnasium's
  RNG) are not seeded anywhere, so Q-learning results (and therefore Return/steps/precision
  numbers) are not exactly reproducible run-to-run — only distributionally similar. Worth a
  `torch.manual_seed()` + `env.action_space.seed()` at the top of `learnPolicy`/`main()` if
  exact reproducibility matters for the paper's numbers.
- **Hardcoded Windows absolute paths** (`F:\Thesis Supervised\Year 2023\...`) are still
  present in `Read_Amazon.py`, `Amazon_RLRecommender1.py`, `ML100K_RLRecommender1.py` (see
  section C) — anyone else cloning this repo hits the same dead-path crashes I fixed
  elsewhere.
- `.venv/` is correctly excluded via its own auto-generated `.gitignore`, and `__pycache__/`
  was untracked in the last commit — good, no action needed there.

---

## Prioritized improvement plan

**P0 — correctness, before trusting any more reported numbers**
1. Decide intent of A1 (subsetcount/goal convergence) — either wire `sscount` back into
   `env.subsetcount` or delete the dead parameter/plumbing and document that termination is
   revisit-count/step-cap based.
2. Fix A2 (`strt` episode-reset bug) — reset per-episode state correctly.
3. Fix A3 (stale `reward` on no-op boundary actions) — initialize `reward = 0` at the top of
   each `while True:` iteration.
4. Fix A4 (module-level accumulator lists) — move into `run()`'s local scope.

**P1 — reproducibility, before anyone else tries to run this**
5. Add `requirements.txt` (versions already known from this session's install).
6. Fix remaining dead `F:\...` paths in `Read_Amazon.py` and both `*_RLRecommender1.py`;
   guard those two files with `if __name__ == '__main__':` so they're safely importable.
7. Seed `torch`/`gymnasium` RNGs for exact repeatability.
8. Reconcile `MultiUserEval.py` (40-user sample) vs. the fixed-up `*_RLRecommender1.py`
   (full-population loop) — pick one as canonical for the paper's reported numbers, or
   clearly label both (e.g. "full-population" figure vs. "held-out sample" table).

**P2 — cleanup, low-risk anytime**
9. Delete dead code from section B (Triangle.py, computeSim, computeGoal2/3,
   computeCoverage, actions dict, Listcount.py's pasted data, unused vars, ListPlot.py or
   move it to a clearly-labeled `scripts/`/`archive/` folder if it's meant to be kept for
   provenance).
10. Rename `computeCoverage2` → something like `compute_recommendation_novelty` to match
    what it actually measures, and update the paper's terminology accordingly if "coverage"
    is used in Section 4.

**P3 — structural, do only if there's appetite for a bigger refactor (separate from the paper revision)**
11. Collapse `AmazonExp1.py`/`ML100KExp1.py`, `StartState2.py`/`ML100K_StartState2.py`, and
    `Amazon_KMeansClustering.py`/`ML100K_KMeansClustering.py` pairs into single
    dataset-parameterized modules.
12. Extract the shared step/reward/goal-tracking loop out of
    `learnPolicy`/`extractPolicy`/`applyPolicy` into one helper.
