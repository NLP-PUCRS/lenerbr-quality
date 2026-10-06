import json
from pathlib import Path
import pandas as pd
from datetime import timedelta, datetime
import numpy as np

def parse_results_to_csv(results_path: str = "results", output_filename: str = "overall.csv") -> None:
    """Parses individual task summary JSONs and aggregates them into a single CSV grouped by model and time."""
    results = Path(results_path)
    results.mkdir(parents=True, exist_ok=True)
    MAX_DIFF = timedelta(minutes=0)
    rows = []

    # Iterate through all summary files in the results directory
    for filename in results.rglob("sum*.json"):
        with open(filename, "r") as f:
            summary = json.load(f)
        
        # Handle different timestamp formats
        try:
            timestamp = datetime.strptime(summary.get("timestamp"), "%d/%m/%y - %H:%M:%S")
        except ValueError:
            timestamp = datetime.strptime(summary.get("timestamp"), "%d/%m/%Y - %H:%M:%S")

        rows.append({
            "task": summary.get("task"),
            "model": summary.get("run_name"),
            "dataset": summary.get("dataset"),
            "timestamp": timestamp,
            "metrics": summary.get("mean_metrics")
        })
    
    if not rows:
        print("No results found.")
        return

    df_raw = pd.DataFrame(rows)
    final_rows = []

    # Group tasks belonging to the same experiment run (proximity in time)
    for model, group in df_raw.groupby("model"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        used = set()
        
        for i, row in group.iterrows():
            # Use NER task as the anchor for the experiment run
            if i in used or not row["task"].startswith("ner"):
                continue
            
            base = {
                "model": model,
                "dataset": row.get("dataset"),
                "timestamp": row["timestamp"]
            }
            used.add(i)
            
            # Find associated STS and RTE tasks within the time window
            for j, other in group.iterrows():
                if j in used:
                    continue
                
                if abs(other["timestamp"] - row["timestamp"]) <= MAX_DIFF:
                    task = other["task"]
                    metrics = other["metrics"]
                    
                    if task == "sts":
                        base.update({
                            "sts_mse": metrics.get("mean_mse"),
                            "sts_mse_std": metrics.get("std_mse"),
                            "sts_pearson": metrics.get("mean_pearson"),
                            "sts_pearson_std": metrics.get("std_pearson")
                        })
                        used.add(j)
                    elif task == "rte":
                        base.update({
                            "rte_f1": metrics.get("mean_f1_macro"),
                            "rte_f1_std": metrics.get("std_f1_macro"),
                            "rte_accuracy": metrics.get("mean_accuracy"),
                            "rte_accuracy_std": metrics.get("std_accuracy")
                        })
                        used.add(j)
            
            # Add NER overall metrics to the base entry
            ner_metrics = row["metrics"]
            base.update({
                "ner_f1": ner_metrics.get("mean_f1"),
                "ner_f1_std": ner_metrics.get("std_f1"),
                "ner_recall": ner_metrics.get("mean_recall"),
                "ner_recall_std": ner_metrics.get("std_recall"),
                "ner_precision": ner_metrics.get("mean_precision"),
                "ner_precision_std": ner_metrics.get("std_precision"),
                "ner_accuracy": ner_metrics.get("mean_accuracy"),
                "ner_accuracy_std": ner_metrics.get("std_accuracy")
            })

            # Add per-entity class metrics (e.g. mean_PESSOA_f1, std_PESSOA_f1, ...)
            per_class_keys = [
                k for k in ner_metrics
                if k.startswith("mean_") and k.endswith(("_f1", "_precision", "_recall"))
                and not k in ("mean_f1", "mean_precision", "mean_recall")
            ]
            for mean_key in per_class_keys:
                std_key = mean_key.replace("mean_", "std_", 1)
                col = f"ner_{mean_key[len('mean_'):]}"
                base[col] = ner_metrics.get(mean_key)
                base[f"{col}_std"] = ner_metrics.get(std_key)
            final_rows.append(base)

    # Save the aggregated results to CSV
    df_final = pd.DataFrame(final_rows)
    df_final = df_final.sort_values("timestamp")
    df_final.to_csv(results / output_filename, index=False)

if __name__ == "__main__":
    parse_results_to_csv()
