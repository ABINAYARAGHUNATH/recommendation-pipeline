import pandas as pd
from src.utils import GENRES

def build_features(train, users, items, user_summary, item_pop):
    """One function used for BOTH training and prediction, so features stay consistent."""
    ref = train["timestamp"].max()   # reference time = end of the training period

    # User features
    uf = user_summary.rename(columns={"n_interactions": "user_activity_count",
                                      "avg_rating": "user_avg_rating"}).copy()
    uf["days_since_last_interaction"] = (ref - uf["last_ts"]) / 86400

    joined = train.merge(items[["item_id"] + GENRES], on="item_id")
    pref = (joined.groupby("user_id")[GENRES].sum().idxmax(axis=1)
            .rename("preferred_genre").reset_index())
    uf = (uf.merge(pref, on="user_id")
            .merge(users[["user_id", "age", "gender", "occupation"]], on="user_id", how="left"))

    # Item features
    itf = item_pop.merge(items[["item_id", "title"] + GENRES], on="item_id")

    # Interaction features
    inf = train.assign(recency_days=(ref - train["timestamp"]) / 86400)

    return uf, itf, inf

def run(cfg):
    p = cfg["paths"]
    rd = lambda name: pd.read_parquet(f"{p['processed']}/{name}.parquet")

    uf, itf, inf = build_features(rd("train"), rd("users"), rd("items"),
                                  rd("user_summary"), rd("item_popularity"))

    uf.to_parquet(f"{p['features']}/user_features.parquet", index=False)
    itf.to_parquet(f"{p['features']}/item_features.parquet", index=False)
    inf.to_parquet(f"{p['features']}/interaction_features.parquet", index=False)

    print(f"User features: {uf.shape} | Item features: {itf.shape} | Interaction features: {inf.shape}")
    print(uf[["user_id", "user_avg_rating", "user_activity_count",
              "days_since_last_interaction", "preferred_genre"]].head())