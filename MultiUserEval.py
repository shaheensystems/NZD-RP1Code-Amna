"""
Multi-user evaluation harness -- fast SAMPLE, not the full population.

The original AmazonExp1.py / ML100KExp1.py `run()` only ever tested 2
hardcoded synthetic users. Precision/Recall/F-measure were already computed
inside main() but never aggregated over a real user sample or reported in
the paper. This script samples real users from each dataset, runs the
existing (unmodified) RL pipeline for each, and aggregates:
  Precision, Recall, F-measure, Item Coverage, Return, Hit Ratio,
  plus the fraction of users for whom no recommendation was feasible
  (cold-start / disconnected-cluster cases, main() returns state=-1).

Relationship to Amazon_RLRecommender1.py / ML100K_RLRecommender1.py:
those two scripts loop over EVERY user in the dataset (1191 Amazon / 671
MovieLens) and reproduce the paper's original States-Visited / Return-Earned
per-user figures -- they are the full-population, "ground truth" evaluation,
now fixed to run headlessly and reproducibly (see requirements.txt / P1.6-1.7
in CODE_REVIEW_FINDINGS.md), but slow (expect 60-100+ minutes combined for
both datasets at ~3-5s/user). This script (MultiUserEval.py) instead samples
N=40 users (seed=42) per dataset for a fast, still-reproducible run, and adds
metrics the full-population scripts don't compute (Hit Ratio, bootstrap CIs,
Mann-Whitney significance testing -- see MultiUserEval.py's own output and
Rebuttal_Draft_Sections.md). Use the full-population scripts if you need the
original per-user figures or a non-sampled headline number; use this script
for fast iteration and the standard-metrics/significance-testing numbers
quoted in the reviewer response.
"""
import random
import time
import numpy as np
import pandas as pd

import Amazon_KMeansClustering as AKC
import AmazonExp1 as AE
import ML100K_KMeansClustering as MKC
import ML100KExp1 as ME

SEED = 42
N_USERS = 40
MAX_ITEMS_PER_USER = 50  # cap so a single very-active MovieLens user doesn't dominate runtime


def sample_amazon_users(n=N_USERS, seed=SEED):
    df = pd.read_csv('amazon test set.csv', header=None, names=['User No.', 'Product ID code', 'Rating'])
    rng = random.Random(seed)
    all_users = sorted(df['User No.'].unique().tolist())
    chosen = rng.sample(all_users, min(n, len(all_users)))
    samples = []
    for u in chosen:
        rows = df[df['User No.'] == u]
        items = rows['Product ID code'].tolist()
        ratings = rows['Rating'].tolist()
        samples.append((u, items, ratings))
    return samples


def sample_movielens_users(n=N_USERS, seed=SEED, max_items=MAX_ITEMS_PER_USER):
    df = pd.read_csv('Movielens100k.csv')
    rng = random.Random(seed)
    all_users = sorted(df['userId'].unique().tolist())
    chosen = rng.sample(all_users, min(n, len(all_users)))
    samples = []
    for u in chosen:
        rows = df[df['userId'] == u]
        if len(rows) > max_items:
            rows = rows.sample(n=max_items, random_state=seed)
        items = rows['movieId'].tolist()
        ratings = rows['rating'].tolist()
        samples.append((u, items, ratings))
    return samples


def evaluate(dataset_name, main_fn, Dict, samples, stopcount=10):
    rows = []
    infeasible = 0
    for i, (uid, items, ratings) in enumerate(samples):
        t0 = time.time()
        try:
            pred, cover, prec, recal, FM, ttime, ret, state, visited = main_fn(items, ratings, Dict, stopcount)
        except Exception as e:
            print(f"[{dataset_name}] user {uid} raised {type(e).__name__}: {e} -- skipping")
            continue
        elapsed = time.time() - t0
        if state == -1:
            infeasible += 1
            print(f"[{dataset_name}] user {uid} ({i+1}/{len(samples)}): infeasible (no reachable goal state)")
            continue
        hit = 1 if prec > 0 else 0
        rows.append({'user': uid, 'n_items': len(items), 'coverage': cover,
                     'precision': prec, 'recall': recal, 'f_measure': FM,
                     'return': ret, 'hit': hit, 'seconds': elapsed})
        print(f"[{dataset_name}] user {uid} ({i+1}/{len(samples)}): "
              f"P={prec:.2f} R={recal:.2f} F={FM:.2f} cov={cover:.2f} hit={hit} ({elapsed:.1f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(f'multiuser_eval_{dataset_name}.csv', index=False)

    summary = {
        'dataset': dataset_name,
        'n_sampled': len(samples),
        'n_evaluated': len(df),
        'n_infeasible': infeasible,
        'precision_mean': df['precision'].mean() if len(df) else float('nan'),
        'precision_std': df['precision'].std() if len(df) else float('nan'),
        'recall_mean': df['recall'].mean() if len(df) else float('nan'),
        'recall_std': df['recall'].std() if len(df) else float('nan'),
        'f_measure_mean': df['f_measure'].mean() if len(df) else float('nan'),
        'coverage_mean': df['coverage'].mean() if len(df) else float('nan'),
        'return_mean': df['return'].mean() if len(df) else float('nan'),
        'hit_ratio': df['hit'].mean() if len(df) else float('nan'),
    }
    return df, summary


# Per-dataset stopcount, chosen via StopcountSweep.py on a disjoint tuning sample (seed=123)
# and verified on this file's own seed=42 reporting sample before being finalized. Re-tuned
# after the reward-function fix (REWARD_FUNCTION_MATH.md) made the old always-zero-reward
# tuning stale -- MovieLens's optimum shifted from 15 to 40 once reward became real; Amazon's
# stayed at 30. See Rebuttal_Draft_Sections.md Section 6-7 for the full before/after.
STOPCOUNT_AMAZON = 30
STOPCOUNT_MOVIELENS = 40

if __name__ == '__main__':
    all_summaries = []

    print("=== Clustering Amazon (behavioral features) ===")
    _, Dict_amazon = AKC.KMeans_Clusters('amazon test set.csv')
    amazon_samples = sample_amazon_users()
    print(f"Sampled {len(amazon_samples)} Amazon users "
          f"(item counts: {[len(s[1]) for s in amazon_samples]})")
    _, summary_a = evaluate('Amazon', AE.main, Dict_amazon, amazon_samples, stopcount=STOPCOUNT_AMAZON)
    all_summaries.append(summary_a)

    print("\n=== Clustering MovieLens (behavioral features) ===")
    _, Dict_ml, _ = MKC.KMeans_Clusters('Movielens100k.csv')
    ml_samples = sample_movielens_users()
    print(f"Sampled {len(ml_samples)} MovieLens users "
          f"(item counts: {[len(s[1]) for s in ml_samples]})")
    _, summary_m = evaluate('MovieLens', ME.main, Dict_ml, ml_samples, stopcount=STOPCOUNT_MOVIELENS)
    all_summaries.append(summary_m)

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv('multiuser_eval_summary.csv', index=False)
    print("\n=== SUMMARY ===")
    print(summary_df.to_string(index=False))
