from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    DataCollatorForTokenClassification,
)
from config import MODELS_DICT, DATASETS_DICT
from data_utils import load_ner_dataset, preprocess_ner
from metrics import get_compute_metrics_ner
from trainer_utils import train_and_evaluate
from datetime import datetime
from dotenv import load_dotenv

def main() -> None:
    """Entry point to execute fine-tuning and evaluation on NER for all configured models and datasets."""
    load_dotenv()

    results = {}

    # Iterate through datasets defined in config.py
    for dataset_name, dataset_spec in DATASETS_DICT.items():
        print(f"Loading dataset: {dataset_name}")
        ner_ds, label_list = load_ner_dataset(dataset_spec)
        num_ner_labels = len(label_list)
        ner_compute_metrics_fn = get_compute_metrics_ner(label_list)

        results[dataset_name] = {}

        # Iterate through models defined in config.py
        for run_name, model_name in MODELS_DICT.items():
            print(f"Executing: {run_name} ({model_name}) on {dataset_name}")
            timestamp = datetime.now()
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Phase 1: NER Task
            print("Preparing NER data")
            tokenized_ner = ner_ds.map(
                lambda x: preprocess_ner(x, tokenizer),
                batched=True,
                remove_columns=ner_ds["train"].column_names
            )
            tokenized_ner = tokenized_ner.filter(lambda x: len(x['input_ids']) > 0)
            data_collator_ner = DataCollatorForTokenClassification(tokenizer)

            def model_init_ner():
                return AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_ner_labels)

            ner_metrics = train_and_evaluate(
                run_name=run_name,
                task_name=f"ner_{dataset_name}",
                timestamp=timestamp,
                model_init_fn=model_init_ner,
                train_ds=tokenized_ner["train"],
                eval_ds=tokenized_ner["validation"],
                test_ds=tokenized_ner["test"],
                compute_metrics_fn=ner_compute_metrics_fn,
                metric_name="f1",
                data_collator=data_collator_ner,
                direction="maximize",
                gradient_checkpointing=False,
                dataset_name=dataset_name
            )

            results[dataset_name][run_name] = {
                "model": model_name,
                "NER": ner_metrics,
            }

    # Print summary of results
    for dataset_name, model_results in results.items():
        print(f"\nDataset: {dataset_name}")
        for model, metrics in model_results.items():
            print(f"{model}")
            print("NER F1:", f"{metrics['NER'].get('mean_f1', 0):.4f}")
            print("NER Accuracy:", f"{metrics['NER'].get('mean_accuracy', 0):.4f}")

if __name__ == "__main__":
    main()
