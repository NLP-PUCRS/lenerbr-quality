import numpy as np
import evaluate
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_recall_fscore_support
)
from scipy.stats import pearsonr, spearmanr
from transformers import EvalPrediction
from typing import Callable

# Load seqeval for NER evaluation
seqeval_metric = evaluate.load("seqeval")

def compute_metrics_rte(eval_pred: EvalPrediction) -> dict:
    """Calculates macro-averaged F1, accuracy, precision, and recall for RTE."""
    preds, labels = eval_pred
    # Get the predicted class with highest probability
    preds = preds.argmax(axis=1)
    
    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)

    return {
        "f1_macro": f1,
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
    }

def compute_metrics_sts(eval_pred: EvalPrediction) -> dict:
    """Calculates MSE, Pearson, Spearman, and MAE for STS regression."""
    preds, labels = eval_pred
    preds = preds.squeeze()
    
    return {
        "mse": mean_squared_error(labels, preds),
        "pearson": pearsonr(preds, labels)[0],
        "spearman": spearmanr(preds, labels)[0], 
        "mae": mean_absolute_error(labels, preds)
    }

def get_compute_metrics_ner(label_list: list[str]) -> Callable[[EvalPrediction], dict]:
    """Returns a closure that computes NER metrics using a specific label mapping."""

    def compute_metrics_ner(eval_pred: EvalPrediction) -> dict:
        """Function to calculate precision, recall, F1, and accuracy for NER."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        # Filter out padding and special tokens marked with -100
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)

        metrics = {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

        # Add per-entity class metrics (seqeval returns a dict per class)
        for key, val in results.items():
            if isinstance(val, dict):
                metrics[f"{key}_f1"] = val["f1"]
                metrics[f"{key}_precision"] = val["precision"]
                metrics[f"{key}_recall"] = val["recall"]

        return metrics
    return compute_metrics_ner