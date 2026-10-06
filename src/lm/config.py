import os
import json
import optuna
from dotenv import load_dotenv
load_dotenv()

models_env = os.getenv("MODELS_DICT", '{"mmBERT":"jhu-clsp/mmbert-base"}')
MODELS_DICT = json.loads(models_env)

# Maps a dataset run-name to how it should be loaded, e.g.:
# {"lener_br": {"type": "hub", "path": "peluz/lener_br", "revision": "refs/convert/parquet"},
#  "filtered": {"type": "disk", "path": "../data/leNER-Br_filtered/"}}
datasets_env = os.getenv(
    "DATASETS_DICT",
    '{"lener_br": {"type": "hub", "path": "peluz/lener_br", "revision": "refs/convert/parquet"}, "filtered": {"type": "disk", "path": "data/leNER-Br_filtered/"}}'
)
DATASETS_DICT = json.loads(datasets_env)

seeds_env = os.getenv("SEEDS_LIST", "42,1903,13,50")
SEEDS_LIST = [int(seed.strip()) for seed in seeds_env.split(",")]

N_OPTUNA_TRIALS = int(os.getenv("N_OPTUNA_TRIALS", "5"))

def optuna_hp_space(trial:optuna.Trial) -> dict:
    """Defines the hyperparameter search space using optuna."""

    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 30)
    }