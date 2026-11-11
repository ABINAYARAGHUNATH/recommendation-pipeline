import joblib
import mlflow
import pandas as pd
from src.training.models import PopularityRecommender, SVDRecommender

def run(cfg, name):
    p = cfg["paths"]
    train = pd.read_parquet(f"{p['processed']}/train.parquet")

    if name == "popularity":
        model = PopularityRecommender()
        params = {"model": name}
    else:
        model = SVDRecommender(cfg["n_factors"], cfg["seed"])
        params = {"model": name, "n_factors": cfg["n_factors"], "seed": cfg["seed"]}
    params["train_rows"] = len(train)

    mlflow.set_experiment(cfg["mlflow_experiment"])
    with mlflow.start_run(run_name=name) as r:
        mlflow.log_params(params)
        model.fit(train)
        path = f"{p['models']}/{name}.pkl"
        joblib.dump(model, path)
        mlflow.log_artifact(path)
        open(f"{p['models']}/{name}_run_id.txt", "w").write(r.info.run_id)
    print(f"Trained {name} -> {path}")