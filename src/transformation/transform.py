import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

def build_matrix(train):
    """Build a sparse user-item rating matrix. Reused later by the models."""
    users = np.sort(train["user_id"].unique())
    items = np.sort(train["item_id"].unique())
    rows = np.searchsorted(users, train["user_id"])
    cols = np.searchsorted(items, train["item_id"])
    mat = csr_matrix((train["rating"].astype(float), (rows, cols)),
                     shape=(len(users), len(items)))
    return mat, users, items

def run(cfg):
    p = cfg["paths"]
    train = pd.read_parquet(f"{p['processed']}/train.parquet")   # train only, so no test leakage

    # 1. User activity summary
    user_summary = (train.groupby("user_id")
                    .agg(n_interactions=("rating", "count"),
                         avg_rating=("rating", "mean"),
                         last_ts=("timestamp", "max"))
                    .reset_index())
    user_summary.to_parquet(f"{p['processed']}/user_summary.parquet", index=False)

    # 2. Item popularity
    item_pop = (train.groupby("item_id")
                .agg(popularity=("rating", "count"),
                     item_avg_rating=("rating", "mean"))
                .reset_index())
    item_pop.to_parquet(f"{p['processed']}/item_popularity.parquet", index=False)

    # 3. User-item interaction matrix
    mat, users, items = build_matrix(train)
    save_npz(f"{p['processed']}/interaction_matrix.npz", mat)

    density = mat.nnz / (mat.shape[0] * mat.shape[1])
    print(f"User summary: {len(user_summary)} users")
    print(f"Item popularity: {len(item_pop)} items")
    print(f"Matrix: {mat.shape[0]} users x {mat.shape[1]} items, density {density:.2%}")