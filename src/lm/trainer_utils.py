import json
import numpy as np
import gc
import shutil
from pathlib import Path
from datetime import datetime
from transformers import TrainingArguments, Trainer, set_seed, DataCollator
from transformers.trainer_callback import EarlyStoppingCallback
import matplotlib.pyplot as plt
from config import SEEDS_LIST, N_OPTUNA_TRIALS, optuna_hp_space
from typing import Any, Callable

def train_and_evaluate(
    run_name: str,
    task_name: str,
    timestamp: datetime,
    model_init_fn: Callable[[], Any],
    train_ds: Any,
    eval_ds: Any,
    test_ds: Any,
    compute_metrics_fn: Callable[[Any], dict],
    metric_name: str,
    data_collator: DataCollator = None,
    direction: str = "maximize",
    gradient_checkpointing: bool = False,
    dataset_name: str = None
) -> dict:
    """Orchestrates hyperparameter search with Optuna and multiple-seed training for a specific task."""
    
    # Run Optuna to find best hyperparameters
    print(f"\n Optuna ({N_OPTUNA_TRIALS} trials) for {run_name} ({task_name})")
    
    tmp_dir = f"./tmp_optuna_{task_name}_{run_name}"
    
    training_args_search = TrainingArguments(
        output_dir=tmp_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        greater_is_better=(direction == "maximize"),
        bf16=True,
        disable_tqdm=True,
        gradient_checkpointing=gradient_checkpointing
    )

    trainer_search = Trainer(
        model=None, 
        model_init=model_init_fn, 
        args=training_args_search,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    best_run = trainer_search.hyperparameter_search(
        direction=direction, 
        backend="optuna", 
        hp_space=optuna_hp_space, 
        n_trials=N_OPTUNA_TRIALS
    )
    
    best_hps = best_run.hyperparameters
    print(f"Best parameters for {run_name} ({task_name}): {best_hps}\n")

    # Cleanup temporary Optuna directory
    if Path(tmp_dir).exists():
        shutil.rmtree(tmp_dir)

    all_seed_results = []
    run_date = timestamp.strftime("%d%m%y_%H%M")
    
    base_out_dir = Path(f"./results/{task_name}")
    base_out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(f"./models/{task_name}")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_total_params = None
    model_trainable_params = None

    # Train and evaluate using multiple seeds for robustness
    for seed in SEEDS_LIST:
        output_dir = base_out_dir / f"ft_{run_name}_seed{seed}_{run_date}"
        print(f"Training {run_name} ({task_name.upper()} | Seed: {seed}) - Output: {output_dir}")
        
        set_seed(seed)

        final_args = TrainingArguments(
            output_dir=str(output_dir),
            logging_dir=str(output_dir / "logs"),
            logging_strategy="epoch",
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model=metric_name,
            greater_is_better=(direction == "maximize"),
            bf16=True,
            seed=seed,
            gradient_checkpointing=gradient_checkpointing,
            learning_rate=best_hps["learning_rate"],
            per_device_train_batch_size=best_hps["per_device_train_batch_size"],
            per_device_eval_batch_size=32,
            weight_decay=best_hps["weight_decay"],
            num_train_epochs=best_hps["num_train_epochs"],
        )

        trainer = Trainer(
            model=None, 
            model_init=model_init_fn, 
            args=final_args,
            train_dataset=train_ds, 
            eval_dataset=eval_ds,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        train_result = trainer.train() 
        eval_metrics = trainer.evaluate(eval_dataset=test_ds)

        # Plot loss history
        log_history = trainer.state.log_history
        
        train_loss = [x['loss'] for x in log_history if 'loss' in x and 'epoch' in x]
        train_epochs = [x['epoch'] for x in log_history if 'loss' in x and 'epoch' in x]
        
        eval_loss = [x['eval_loss'] for x in log_history if 'eval_loss' in x and 'epoch' in x]
        eval_epochs = [x['epoch'] for x in log_history if 'eval_loss' in x and 'epoch' in x]
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_epochs, train_loss, label='Training Loss')
        if eval_epochs and eval_loss:
            plt.plot(eval_epochs, eval_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss Curve - {run_name} ({task_name}) - Seed {seed}')
        plt.legend()
        plt.grid(True)
        plt.savefig(str(output_dir / "loss_curve.png"))
        plt.close()

        # Save the fine-tuned model for this seed
        trainer.save_model(str(model_dir/f"ft_{run_name}_seed{seed}_{run_date}"))

        if model_total_params is None:
            model_total_params = trainer.model.num_parameters()
            model_trainable_params = trainer.model.num_parameters(only_trainable=True)

        seed_metrics = {"seed": seed}

        # Track total floating point operations if available
        if "total_flos" in train_result.metrics:
            seed_metrics["total_flos"] = train_result.metrics["total_flos"]

        for key, value in eval_metrics.items():
            if key.startswith("eval_") and key not in ["eval_loss", "eval_runtime", "eval_samples_per_second", "eval_steps_per_second"]:
                clean_key = key.replace("eval_", "")
                seed_metrics[clean_key] = value
        
        all_seed_results.append(seed_metrics)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "best_hps_used.json", "w") as f: 
            json.dump(best_hps, f, indent=4)
            
        del trainer
        gc.collect()

    # Calculate mean and standard deviation across all seeds
    final_stats = {}
    metric_keys = [k for k in all_seed_results[0].keys() if k != "seed"]
    for key in metric_keys:
        values = [res[key] for res in all_seed_results]
        final_stats[f"mean_{key}"] = np.mean(values)
        final_stats[f"std_{key}"] = np.std(values)

    # Export a comprehensive summary JSON
    summary_path = base_out_dir / f"summary_{run_name}_{run_date}.json"
    summary_data = {
        "task": task_name,
        "run_name": run_name,
        "dataset": dataset_name,
        "timestamp":timestamp.strftime("%d/%m/%y - %H:%M:%S"),
        "model_total_parameters": model_total_params,
        "model_trainable_parameters": model_trainable_params,
        "best_hps": best_hps,
        "seeds_detail": all_seed_results,
        "mean_metrics": final_stats
    }
    with open(summary_path, "w") as f: 
        json.dump(summary_data, f, indent=4)

    return final_stats