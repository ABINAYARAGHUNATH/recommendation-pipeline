import numpy as np
from sklearn.decomposition import TruncatedSVD
from src.transformation.transform import build_matrix

class PopularityRecommender:
    """Baseline: recommend the most-rated items the user hasn't seen."""
    def fit(self, train):
        g = train.groupby("item_id")["rating"].agg(["count", "mean"])
        self.rank = g.sort_values("count", ascending=False).index.to_numpy()
        self.item_mean = g["mean"]
        self.gmean = train["rating"].mean()
        self.seen = train.groupby("user_id")["item_id"].apply(set).to_dict()
        return self

    def recommend(self, user_id, k):
        seen = self.seen.get(user_id, set())
        return [int(i) for i in self.rank if i not in seen][:k]

    def predict(self, df):
        return df["item_id"].map(self.item_mean).fillna(self.gmean).to_numpy()


class SVDRecommender:
    """Personalized: matrix factorization (truncated SVD on mean-centered ratings)."""
    def __init__(self, n_factors=20, seed=42):
        self.n_factors, self.seed = n_factors, seed

    def fit(self, train):
        mat, self.users, self.items = build_matrix(train)
        nnz = np.diff(mat.indptr)
        self.user_mean = np.asarray(mat.sum(axis=1)).ravel() / np.maximum(nnz, 1)
        mat.data -= np.repeat(self.user_mean, nnz)          # mean-center per user
        svd = TruncatedSVD(self.n_factors, random_state=self.seed)
        U = svd.fit_transform(mat)
        self.scores = U @ svd.components_ + self.user_mean[:, None]
        self.u_idx = {u: i for i, u in enumerate(self.users)}
        self.i_idx = {i: j for j, i in enumerate(self.items)}
        self.seen = train.groupby("user_id")["item_id"].apply(set).to_dict()
        return self

    def recommend(self, user_id, k):
        if user_id not in self.u_idx:
            return []
        s = self.scores[self.u_idx[user_id]].copy()
        s[[self.i_idx[i] for i in self.seen[user_id]]] = -np.inf   # hide already-seen items
        top = np.argpartition(-s, k)[:k]
        return self.items[top[np.argsort(-s[top])]].tolist()

    def predict(self, df):
        out = [self.scores[self.u_idx[u], self.i_idx[i]]
               if u in self.u_idx and i in self.i_idx else np.nan
               for u, i in zip(df["user_id"], df["item_id"])]
        return np.clip(out, 1, 5)