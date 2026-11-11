import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_distances
from src.utils import GENRES

def ndcg_at_k(recs, relevant, k):
    dcg = sum(1 / np.log2(i + 2) for i, x in enumerate(recs[:k]) if x in relevant)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg else 0.0

def list_diversity(recs, genre_matrix):
    """Mean pairwise genre dissimilarity inside one recommendation list."""
    n = len(recs)
    d = cosine_distances(genre_matrix.loc[recs].values)
    return d.sum() / (n * (n - 1))

def run(cfg):
    p, k = cfg["paths"], cfg["k"]
    test = pd.read_parquet(f"{p['processed']}/test.parquet")
    items = pd.read_parquet(f"{p['processed']}/items.parquet")
    genres = items.set_index("item_id")[GENRES]

    # Relevant = test items the user rated at or above the threshold
    relevant = (test[test["rating"] >= cfg["relevance_threshold"]]
                .groupby("user_id")["item_id"].apply(set).to_dict())

    rows = []
    for name in ["popularity", "svd"]:
        model = joblib.load(f"{p['models']}/{name}.pkl")
        recs = {u: model.recommend(u, k) for u in relevant}
        hits = {u: len(set(r) & relevant[u]) for u, r in recs.items()}
        pred = model.predict(test)

        m = {
            "model": name,
            f"precision@{k}": np.mean([hits[u] / k for u in recs]),
            f"recall@{k}": np.mean([hits[u] / len(relevant[u]) for u in recs]),
            f"ndcg@{k}": np.mean([ndcg_at_k(r, relevant[u], k) for u, r in recs.items()]),
            "coverage": len(set().union(*map(set, recs.values()))) / len(items),
            "diversity": np.mean([list_diversity(r, genres) for r in recs.values() if len(r) > 1]),
            "rmse": float(np.sqrt(mean_squared_error(test["rating"], pred))),
            "mae": float(mean_absolute_error(test["rating"], pred)),
        }

        # Log metrics to the same MLflow run created during training
        run_id = open(f"{p['models']}/{name}_run_id.txt").read().strip()
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({a.replace("@", "_at_"): float(b)
                                for a, b in m.items() if a != "model"})
        rows.append(m)

    results = pd.DataFrame(rows).round(4)
    results.to_csv(f"{p['reports']}/results.csv", index=False)
    open(f"{p['reports']}/results.md", "w").write(results.to_markdown(index=False))
    print(results.to_string(index=False))