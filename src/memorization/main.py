import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, classification_report, f1_score

from span_classifier import Naive_Span_Classifier
from token_classifier import Naive_Token_Classifier

CLASSIFIERS = {
    "token": Naive_Token_Classifier,
    "span": Naive_Span_Classifier,
}


def load_ner_dataset(dataset_spec: dict) -> tuple[DatasetDict, list[str]]:
    """Loads a LeNER-Br NER dataset either from the Hugging Face Hub or from local disk, per dataset_spec."""
    load_type = dataset_spec.get("type", "hub")
    path = dataset_spec["path"]

    if load_type == "disk":
        raw_ner = load_from_disk(path)
    else:
        raw_ner = load_dataset(path, revision=dataset_spec.get("revision"))

    ner_ds = DatasetDict({
        "train": raw_ner["train"],
        "validation": raw_ner["validation"],
        "test": raw_ner["test"],
    })
    label_list = ner_ds["train"].features["ner_tags"].feature.names
    return ner_ds, label_list


def evaluate_classifier(model, test_set, label_names: list[str]) -> dict:
    """Runs prediction and computes overall + per-entity-class (B-/I- merged) metrics.

    Mirrors the metrics calculated in notebooks/naive_classifier_usage_demo.ipynb.
    """
    y_pred = model.predict(test_set)
    y_true = {register["id"]: register["ner_tags"] for register in test_set}

    y_true_flat, y_pred_flat = [], []
    for id_ in y_pred:
        y_true_flat.extend(y_true[id_])
        y_pred_flat.extend(y_pred[id_])

    def tag_name(tag):
        return label_names[tag] if tag != -1 else "OOV"

    y_true_named = [tag_name(t) for t in y_true_flat]
    y_pred_named = [tag_name(t) for t in y_pred_flat]

    metrics = {
        "accuracy": accuracy_score(y_true_flat, y_pred_flat),
        "f1": f1_score(y_true_flat, y_pred_flat, average="macro"),
    }

    report = classification_report(
        y_true_named, y_pred_named, digits=4, zero_division=0, output_dict=True
    )

    per_class = {}
    for label, scores in report.items():
        if label in ("accuracy", "macro avg", "weighted avg", "O", "OOV"):
            continue
        prefix, entity_class = label.split("-", 1)
        per_class.setdefault(entity_class, {})[prefix] = scores

    for entity_class, prefixes in per_class.items():
        b, i = prefixes.get("B", {}), prefixes.get("I", {})
        metrics[f"{entity_class}_precision"] = (b.get("precision", 0) + i.get("precision", 0)) / 2
        metrics[f"{entity_class}_recall"] = (b.get("recall", 0) + i.get("recall", 0)) / 2
        metrics[f"{entity_class}_f1"] = (b.get("f1-score", 0) + i.get("f1-score", 0)) / 2
        metrics[f"{entity_class}_support_B"] = b.get("support", 0)
        metrics[f"{entity_class}_support_I"] = i.get("support", 0)

    return metrics


def train_and_evaluate(level: str, dataset_name: str, dataset_spec: dict, results_dir: Path) -> dict:
    print(f"Loading dataset: {dataset_name}")
    ner_ds, label_names = load_ner_dataset(dataset_spec)

    run_name = f"rule_based_{level}"
    task_name = f"ner_{dataset_name}"

    print(f"Training {run_name} on {dataset_name}")
    timestamp = datetime.now()
    model = CLASSIFIERS[level]()
    model.train(ner_ds["train"])

    metrics = evaluate_classifier(model, ner_ds["test"], label_names)
    # Prefixed with "mean_" to stay compatible with the fine-tuning summary schema, so both are aggregated by src/result_parsing.py under the same CSV columns.
    mean_metrics = {f"mean_{key}": value for key, value in metrics.items()}

    run_date = timestamp.strftime("%d%m%y_%H%M")
    task_dir = results_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    summary_path = task_dir / f"summary_{run_name}_{run_date}.json"
    summary_data = {
        "task": task_name,
        "run_name": run_name,
        "dataset": dataset_name,
        "timestamp": timestamp.strftime("%d/%m/%y - %H:%M:%S"),
        "deterministic": True,
        "mean_metrics": mean_metrics,
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=4)

    return mean_metrics


def main() -> None:
    """Train and evaluate a rule-based (memorization) NER classifier for all configured datasets."""
    load_dotenv()

    parser = argparse.ArgumentParser(description="Train and evaluate rule-based NER classifiers.")
    parser.add_argument(
        "--level",
        choices=["token", "span", "both"],
        default="token",
        help="Granularity of the memorization classifier to train (default: token).",
    )
    args = parser.parse_args()
    levels = list(CLASSIFIERS) if args.level == "both" else [args.level]

    datasets_env = os.getenv(
        "DATASETS_DICT",
        '{"lener_br": {"type": "hub", "path": "peluz/lener_br", "revision": "refs/convert/parquet"}}',
    )
    datasets_dict = json.loads(datasets_env)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, dataset_spec in datasets_dict.items():
        for level in levels:
            metrics = train_and_evaluate(level, dataset_name, dataset_spec, results_dir)
            print(f"\nDataset: {dataset_name}, Classifier: {level}")
            print("F1 (macro):", f"{metrics.get('mean_f1', 0):.4f}")
            print("Accuracy:", f"{metrics.get('mean_accuracy', 0):.4f}")


if __name__ == "__main__":
    main()
