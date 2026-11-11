import json
import pandas as pd

def validate_interactions(inter, users, items):
    inter = inter.copy()
    for c in ["user_id", "item_id", "rating", "timestamp"]:   # bad types become NaN
        inter[c] = pd.to_numeric(inter[c], errors="coerce")

    reasons = pd.Series("", index=inter.index)
    def flag(mask, why):
        reasons[mask] += why + ";"

    ts = pd.to_datetime(inter["timestamp"], unit="s", errors="coerce")
    flag(inter.isna().any(axis=1), "missing_or_bad_type")
    flag(inter.duplicated(["user_id", "item_id"], keep="first"), "duplicate")
    flag(~inter["rating"].between(1, 5), "invalid_rating")
    flag(ts.isna() | (inter["timestamp"] <= 0) | (ts > pd.Timestamp.now()), "invalid_timestamp")
    flag(~inter["user_id"].isin(users["user_id"]), "orphan_user")
    flag(~inter["item_id"].isin(items["item_id"]), "orphan_item")

    bad = reasons != ""
    return inter[~bad].astype(int), inter[bad].assign(reason=reasons[bad])

def run(cfg):
    p = cfg["paths"]
    users = pd.read_csv(f"{p['raw']}/users.csv")
    items = pd.read_csv(f"{p['raw']}/items.csv")
    inter = pd.read_csv(f"{p['raw']}/interactions.csv")

    good, bad = validate_interactions(inter, users, items)
    good.to_parquet(f"{p['processed']}/interactions_valid.parquet")
    bad.to_csv(f"{p['errors']}/interactions_errors.csv", index=False)

    report = {
        "total_rows": len(inter),
        "valid_rows": len(good),
        "error_rows": len(bad),
        "by_reason": bad["reason"].str.rstrip(";").str.split(";").explode().value_counts().to_dict(),
    }
    json.dump(report, open(f"{p['reports']}/validation_report.json", "w"), indent=2)
    print(report)