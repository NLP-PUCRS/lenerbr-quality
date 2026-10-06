# A Data-Centric Evaluation of the LeNER-Br Dataset

This repository contains the code and experiments for the data-centric evaluation of Brazilian Legal NER (LeNER-Br), as described in the paper.

## ToC

1. [Cite](#cite)
2. [Project Structure](#project-structure)
3. [Dataset](#dataset)
4. [Installation](#installation)
5. [Training Memorization Models](#training-memorization-models)
6. [Fine-tuning LM for NER](#fine-tuning-lm-for-ner)
7. [Results Format](#results-format)

## Cite

- the paper is currently **in press** ([ERAMIA 2026](https://eramia-rs.sbc.org.br/2026/#/)).

```bibtex
@inproceedings{santos2026lenerbrevaluation,
  title={A Data-Centric Evaluation of the LeNER-Br Dataset},
  author={dos Santos, Eduardo Matte Zanardo and do Amaral, Gustavo Losch and Gomes, Carlos Falcão de Azevedo and Griebler, Dalvan},
  booktitle = {Anais da II Escola Regional de Aprendizado de Máquina e Inteligência Artificial da Região Sul},
  location = {Porto Alegre/RS},
  pages = {},
  publisher = {SBC},
  year = {in press}
}
```

## Project Structure

- `src/lm`: Language Model (LM) fine-tuning pipeline, including data utilities, metrics, and trainer orchestration.
- `src/memorization`: Token- and Span-level memorization classifier training and inference scripts.
- `notebooks/`: Exploratory and demonstration notebooks.

## Dataset

We used LeNER-Br, in the [HuggingFace (HF)](https://huggingface.co/datasets/peluz/lener_br) and [GitHub (GH)](https://teodecampos.github.io/LeNER-Br/) versions, to analyse leakage through a data-centric perspective. With it, we trained a memorization classification model and fine-tuned a portuguese adapted ModernBERT-based model.

### Building the Datasets

On top of the HF version, three derived datasets can be built with the `build_*.py` scripts in `src/`. In all of them the test split is kept untouched, and only the train and validation splits are filtered:

- **Sentence Deduped** (`build_sentence_deduped.py`): removes train and validation sentences that are most similar (Jaccard similarity = 1) to sentences of another split.
- **Entity Exemplar Deduped** (`build_entity_exemplar_deduped.py`): built on top of the Sentence Deduped dataset, removes train sentences containing any entity exemplar (entity class + token span) present in validation or test, and validation sentences containing any entity exemplar present in test.
- **Random Downsample** (`build_random_downsample.py`): randomly removes sentences, serving as a size-matched baseline for the deduped versions.

To build them:

```shell
# Sentence Deduped (saved to data/lenerbr_sentence_deduped)
python src/build_sentence_deduped.py

# Entity Exemplar Deduped (saved to data/lenerbr_entity_deduped), must run after the Sentence Deduped
python src/build_entity_exemplar_deduped.py

# Random Downsample (saved to data/lenerbr_random_downsample)
python src/build_random_downsample.py
```

The Entity Exemplar Deduped script also accepts `--source`, `--output` and `--log` arguments, and writes a CSV log of every removed sentence and its matching entity to `results/entity_filtered_removals.csv` by default. To use any of the built datasets in the experiments, add them to `DATASETS_DICT` with `"type": "disk"` (see `.env.example`).

## Installation

Install dependencies using pip:

```shell
pip install -r requirements.txt
```

Copy the `.env.example` file to create your own `.env` file. After copying the file, you will need to change blabla (not ready yet).

```shell
cp .env.example
```

## Training Memorization Models

The memorization models are classifiers that predict NER classes by assigning the most frequent class seen during the iteration on the training split, with no learned generalization. Two granularities are available:

- **Token-level** (`token_classifier.py`): memorizes the most frequent entity class per individual token (used on the paper).
- **Span-level** (`span_classifier.py`): memorizes whole entity spans (possibly multiple tokens).

To train and evaluate them:

```shell
# Token-level classifier (default)
python src/memorization/main.py

# Span-level classifier
python src/memorization/main.py --level span
```

Like the fine-tuning pipeline, it iterates over every dataset defined in `DATASETS_DICT` (see `.env.example`), so a summary is produced per dataset/level combination. Since these classifiers are deterministic (no learned parameters, no seeds), a single run fully determines their metrics.

## Fine-tuning LM for NER

To fine-tune and evaluate models on Portuguese downstream tasks:

1. **Configure Models**: Add your models to the `MODELS_DICT` in `src/lm/config.py`.
2. **Set Hyperparameters**: Adjust the Optuna search space in the `optuna_hp_space` function within the same file.
3. **Run Experiment**:

  ```shell
  # Direct execution
  python src/lm/main.py
  ```

The pipeline automatically performs hyperparameter optimization using Optuna and runs multiple training replications across different seeds.

## Results Format

Upon completion of the experiments, the results are stored in `results/`.

Every summary file is written under `results/<task>/`, where `<task>` is `ner_<dataset_name>` for the dataset it was evaluated on (`<dataset_name>` matching a key in `DATASETS_DICT`).

### Individual Fine-Tuning Summaries

Each fine-tuning of a language model generates a JSON summary file (e.g., `summary_<model>_<timestamp>.json`) that contains model metadata (parameters, name), the dataset it was trained on, best hyperparameters, detailed metrics for each seed replication and aggregated mean and standard deviation for all evaluation metrics across all NER classes.

### Individual memorization Model Summaries

Each memorization classifier run generates a JSON summary file (e.g., `summary_rule_based_<level>_<timestamp>.json`) with the same shape as the fine-tuning summaries (`task`, `run_name`, `dataset`, `timestamp`, `mean_metrics`), using the overall and per-class metrics. Since the classifier is deterministic, `mean_metrics` holds a single `mean_*` value per metric and there are no `std_*` counterparts or `best_hps`/`seeds_detail` fields.

### Aggregated Results

You can use `src/result_parsing.py` to aggregate all individual task summaries (both fine-tuning and memorization) into a single CSV file that contains the model name, the dataset version used for the run, timestamp of experiment execution, and mean and std of each NER class metric over the replication seeds:

```shell
python src/result_parsing.py
```

Rows coming from a memorization model leave every `_std` column (and `ner_precision`/`ner_recall`, which the memorization evaluation does not compute) empty, since there is no replication.
