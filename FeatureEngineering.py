"""
Behavioral feature construction for K-Means clustering.

Replaces the previous approach of clustering on a single label-encoded
ID column (which only bins entities by arbitrary ID order and does not
reflect interaction behavior) with small, dataset-appropriate behavioral
feature vectors, standardized before clustering.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def amazon_features(df, user_col='User No.', product_col='Product ID code', rating_col=None):
    """
    Amazon dataset only exposes User No., Product ID code, and (optionally) Rating
    locally. Behavioral features are therefore rating-statistics based:
      user vector  = [mean rating given, count of ratings given, std of ratings given]
      product vector = [mean rating received, count of ratings received, std of ratings received]
    """
    work = df.copy()
    if rating_col is None or rating_col not in work.columns:
        # No rating column available -> fall back to interaction-count-only features
        user_stats = work.groupby(user_col).size().rename('count').reset_index()
        user_stats['mean'] = 0.0
        user_stats['std'] = 0.0
        prod_stats = work.groupby(product_col).size().rename('count').reset_index()
        prod_stats['mean'] = 0.0
        prod_stats['std'] = 0.0
    else:
        user_stats = work.groupby(user_col)[rating_col].agg(['mean', 'count', 'std']).reset_index()
        prod_stats = work.groupby(product_col)[rating_col].agg(['mean', 'count', 'std']).reset_index()

    user_stats['std'] = user_stats['std'].fillna(0.0)
    prod_stats['std'] = prod_stats['std'].fillna(0.0)

    user_feat_df = work[[user_col]].drop_duplicates().merge(user_stats, on=user_col, how='left')
    prod_feat_df = work[[product_col]].drop_duplicates().merge(prod_stats, on=product_col, how='left')

    X_users = StandardScaler().fit_transform(user_feat_df[['mean', 'count', 'std']].values)
    X_products = StandardScaler().fit_transform(prod_feat_df[['mean', 'count', 'std']].values)

    return X_users, user_feat_df[user_col].values, X_products, prod_feat_df[product_col].values


GENRE_LIST = [
    '(no genres listed)', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX',
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]


def _genre_multihot(genres_str):
    if not isinstance(genres_str, str) or genres_str.strip() == '':
        genres_str = '(no genres listed)'
    tags = set(genres_str.split('|'))
    return np.array([1.0 if g in tags else 0.0 for g in GENRE_LIST])


def movielens_features(df, user_col='userId', product_col='movieId',
                        rating_col='rating', genres_col='genres'):
    """
    MovieLens has real content (genres) and enough ratings/user for a genre-taste
    profile:
      user vector    = [mean rating per genre across genres they rated (0 if unrated),
                         overall mean rating, rating count]
      product vector = [genre multi-hot vector, mean rating received, rating count]
    """
    work = df.copy()
    work['genre_vec'] = work[genres_col].apply(_genre_multihot)

    # ---- product (movie) features: genre multi-hot + rating stats ----
    prod_group = work.groupby(product_col)
    prod_ids = []
    prod_rows = []
    for pid, g in prod_group:
        genre_vec = g['genre_vec'].iloc[0]
        mean_r = g[rating_col].mean()
        count_r = g[rating_col].count()
        prod_ids.append(pid)
        prod_rows.append(np.concatenate([genre_vec, [mean_r, count_r]]))
    X_products = StandardScaler().fit_transform(np.vstack(prod_rows))

    # ---- user features: per-genre average rating (taste profile) ----
    genre_matrix = np.vstack(work['genre_vec'].values)
    ratings = work[rating_col].values.reshape(-1, 1)
    weighted = genre_matrix * ratings
    tmp = pd.DataFrame(weighted, columns=[f'g_{g}' for g in GENRE_LIST])
    tmp['count_mask'] = 1
    genre_count_df = pd.DataFrame(genre_matrix, columns=[f'gc_{g}' for g in GENRE_LIST])
    tmp[user_col] = work[user_col].values
    genre_count_df[user_col] = work[user_col].values

    sum_by_user = tmp.groupby(user_col).sum()
    count_by_user = genre_count_df.groupby(user_col).sum()

    genre_pref = sum_by_user[[f'g_{g}' for g in GENRE_LIST]].values / \
        np.clip(count_by_user[[f'gc_{g}' for g in GENRE_LIST]].values, 1, None)

    overall = work.groupby(user_col)[rating_col].agg(['mean', 'count'])
    user_ids = sum_by_user.index.values
    overall = overall.loc[user_ids]

    X_users_raw = np.concatenate([genre_pref, overall[['mean', 'count']].values], axis=1)
    X_users = StandardScaler().fit_transform(X_users_raw)

    return X_users, user_ids, X_products, np.array(prod_ids)
