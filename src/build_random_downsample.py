# Code to build Random Downsampled dataset on top of HF fetched version of LeNER-Br

import random
from datasets import load_dataset, DatasetDict

def load_datasets() -> tuple[DatasetDict, list[str]]:
    """Loads and splits LeNER-Br and ASSIN2 datasets for NER task."""

    # Load LeNER-Br from Hugging Face
    raw_ner = load_dataset("peluz/lener_br", revision="refs/convert/parquet")
    ner_ds = DatasetDict({
        "train": raw_ner["train"],
        "validation": raw_ner["validation"],
        "test": raw_ner["test"]
    })
    
    # Get the unique list of NER tags
    label_list = ner_ds["train"].features["ner_tags"].feature.names
    
    return ner_ds, label_list

random_dataset, labels = load_datasets()
random.seed(42)
ids_to_remove_train = random.sample(random_dataset["train"]["id"], 1415)
ids_to_remove_val = random.sample(random_dataset["validation"]["id"], 191)
random_dataset["train"] = random_dataset["train"].filter(lambda x: x["id"] not in ids_to_remove_train)
random_dataset["validation"] = random_dataset["validation"].filter(lambda x: x["id"] not in ids_to_remove_val)
random_dataset.save_to_disk("data/lenerbr_random_downsample")