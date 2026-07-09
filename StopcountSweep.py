"""
Hyperparameter sweep for `stopcount` (the revisit-count / convergence threshold in
TwoDGridWorld). This value was a hardcoded magic number (10) with no stated
justification in the original code. Since fixing the dead subsetcount-convergence bug
(CODE_REVIEW_FINDINGS.md P0.1) made this parameter live for the first time, its old
default value was never actually tuned for the post-fix behavior.

Methodology note (avoiding test-set leakage): this sweep uses a TUNING sample (seed=123,
disjoint from the seed=42 sample used throughout Rebuttal_Draft_Sections.md) to pick a
stopcount value. The final reported numbers are then generated on the original seed=42
sample with the chosen value, so the reported metrics are not directly optimized on the
same users they're evaluated on.
"""
import time
import numpy as np
import pandas as pd

import Amazon_KMeansClustering as AKC
import AmazonExp1 as AE
import ML100K_KMeansClustering as MKC
import ML100KExp1 as ME
import MultiUserEval as MUE

STOPCOUNTS = [5, 10, 15, 20, 30]
TUNE_SEED = 123
TUNE_N = 15


def evaluate_stopcount(main_fn, Dict, samples, stopcount):
    rows = []
    for uid, items, ratings in samples:
        try:
            pred, cover, prec, recal, FM, ttime, ret, state, visited = main_fn(items, ratings, Dict, stopcount)
        except Exception as e:
            print(f"  user {uid} raised {type(e).__name__}: {e} -- skipping")
            continue
        if state == -1:
            continue
        hit = 1 if prec > 0 else 0
        rows.append({'precision': prec, 'recall': recal, 'f_measure': FM, 'hit': hit})
    df = pd.DataFrame(rows)
    return {
        'stopcount': stopcount,
        'n': len(df),
        'precision': df['precision'].mean() if len(df) else float('nan'),
        'recall': df['recall'].mean() if len(df) else float('nan'),
        'f_measure': df['f_measure'].mean() if len(df) else float('nan'),
        'hit_ratio': df['hit'].mean() if len(df) else float('nan'),
    }


if __name__ == '__main__':
    print("=== Clustering (shared across all stopcount values) ===")
    _, Dict_amazon = AKC.KMeans_Clusters('amazon test set.csv')
    _, Dict_ml, _ = MKC.KMeans_Clusters('Movielens100k.csv')

    amazon_tune = MUE.sample_amazon_users(n=TUNE_N, seed=TUNE_SEED)
    ml_tune = MUE.sample_movielens_users(n=TUNE_N, seed=TUNE_SEED)

    results = []
    for sc in STOPCOUNTS:
        t0 = time.time()
        ra = evaluate_stopcount(AE.main, Dict_amazon, amazon_tune, sc)
        ra['dataset'] = 'Amazon'
        rm = evaluate_stopcount(ME.main, Dict_ml, ml_tune, sc)
        rm['dataset'] = 'MovieLens'
        results.append(ra)
        results.append(rm)
        print(f"stopcount={sc}: Amazon P={ra['precision']:.3f} R={ra['recall']:.2f} F={ra['f_measure']:.3f} Hit={ra['hit_ratio']:.3f}"
              f"  |  MovieLens P={rm['precision']:.3f} R={rm['recall']:.2f} F={rm['f_measure']:.3f} Hit={rm['hit_ratio']:.3f}"
              f"  ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(results)
    df.to_csv('stopcount_sweep_results.csv', index=False)
    print("\nSaved to stopcount_sweep_results.csv")
