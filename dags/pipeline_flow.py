from prefect import flow, task

from src.utils import load_config
from src.ingestion import ingest
from src.validation import validate
from src.preparation import prepare
from src.transformation import transform
from src.features import build_features
from src.training import train as trainer
from src.evaluation import evaluate

cfg = load_config()

@task(name="ingest", retries=1)
def t_ingest():
    ingest.run(cfg)

@task(name="validate")
def t_validate():
    validate.run(cfg)

@task(name="prepare")
def t_prepare():
    prepare.run(cfg)

@task(name="transform")
def t_transform():
    transform.run(cfg)

@task(name="features")
def t_features():
    build_features.run(cfg)

@task(name="train")
def t_train(model_name):
    trainer.run(cfg, model_name)

@task(name="evaluate")
def t_evaluate():
    evaluate.run(cfg)

@flow(name="recsys-pipeline", log_prints=True)
def pipeline():
    a = t_ingest.submit()
    b = t_validate.submit(wait_for=[a])
    c = t_prepare.submit(wait_for=[b])
    d = t_transform.submit(wait_for=[c])
    e = t_features.submit(wait_for=[d])
    f1 = t_train.submit("popularity", wait_for=[e])   # the two models train in parallel
    f2 = t_train.submit("svd", wait_for=[e])
    t_evaluate.submit(wait_for=[f1, f2]).result()

if __name__ == "__main__":
    pipeline()