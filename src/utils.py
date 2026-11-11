import yaml
from pathlib import Path

GENRES = ["unknown","Action","Adventure","Animation","Children","Comedy","Crime",
          "Documentary","Drama","Fantasy","FilmNoir","Horror","Musical","Mystery",
          "Romance","SciFi","Thriller","War","Western"]

def load_config(path="configs/config.yaml"):
    cfg = yaml.safe_load(open(path))
    for p in cfg["paths"].values():
        Path(p).mkdir(parents=True, exist_ok=True)
    return cfg