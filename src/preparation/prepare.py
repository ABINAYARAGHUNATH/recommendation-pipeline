import pandas as pd

def run(cfg):
    p = cfg["paths"]
    inter = pd.read_parquet(f"{p['processed']}/interactions_valid.parquet")
    users = pd.read_csv(f"{p['raw']}/users.csv").drop_duplicates("user_id")
    items = pd.read_csv(f"{p['raw']}/items.csv").drop_duplicates("item_id")

    # Clean and standardize
    users["age"] = users["age"].fillna(users["age"].median())
    users["gender"] = users["gender"].str.upper()
    items["release_date"] = pd.to_datetime(items["release_date"], errors="coerce")
    inter = inter.drop_duplicates(["user_id", "item_id"]).sort_values("timestamp")

    # Time-based split (NOT random): older 80% for train, newest 20% for test
    cut = inter["timestamp"].quantile(1 - cfg["test_ratio"])
    train = inter[inter["timestamp"] <= cut]
    test = inter[inter["timestamp"] > cut]

    # Keep only test users/items that also appear in train (avoids cold-start in evaluation)
    test = test[test["user_id"].isin(train["user_id"]) & test["item_id"].isin(train["item_id"])]

    for name, df in [("users", users), ("items", items), ("train", train), ("test", test)]:
        df.to_parquet(f"{p['processed']}/{name}.parquet", index=False)

    print(f"Train: {len(train)} rows | Test: {len(test)} rows")
    print("Train ends:", pd.to_datetime(train['timestamp'].max(), unit='s'))
    print("Test starts:", pd.to_datetime(test['timestamp'].min(), unit='s'))