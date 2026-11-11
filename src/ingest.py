import urllib.request, zipfile
import pandas as pd
from pathlib import Path
from src.utils import GENRES

def run(cfg):
    raw = Path(cfg["paths"]["raw"])
    z = raw / "ml-100k.zip"
    if not z.exists():
        urllib.request.urlretrieve(cfg["data_url"], z)
        zipfile.ZipFile(z).extractall(raw)
    base = raw / "ml-100k"
    inter = pd.read_csv(base/"u.data", sep="\t", names=["user_id","item_id","rating","timestamp"])
    users = pd.read_csv(base/"u.user", sep="|", names=["user_id","age","gender","occupation","zip"])
    items = pd.read_csv(base/"u.item", sep="|", encoding="latin-1",
                        names=["item_id","title","release_date","video_release","url"] + GENRES)

    if cfg.get("inject_noise"):
        bad = inter.sample(5, random_state=cfg["seed"]).copy().reset_index(drop=True)
        bad.loc[0, "rating"] = 9
        bad.loc[1, "user_id"] = 99999
        bad.loc[2, "timestamp"] = -5
        inter = pd.concat([inter, bad], ignore_index=True)

    users.to_csv(raw/"users.csv", index=False)
    items.to_csv(raw/"items.csv", index=False)
    inter.to_csv(raw/"interactions.csv", index=False)
    print("Ingested:", len(users), "users,", len(items), "items,", len(inter), "interactions")