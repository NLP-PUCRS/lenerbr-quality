from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer

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
        "test": raw_ner["test"]
    })

    # Get the unique list of NER tags
    label_list = ner_ds["train"].features["ner_tags"].feature.names

    return ner_ds, label_list

def preprocess_ner(examples: dict, tokenizer: AutoTokenizer) -> dict:
    """Tokenizes word sequences and aligns NER tags with the resulting tokens.

    Documents longer than the tokenizer's max length are split into multiple
    overflowing chunks; overflow_to_sample_mapping tracks which original
    example each output chunk came from, since one input can yield several
    outputs.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        return_overflowing_tokens=True
    )
    sample_map = tokenized_inputs.pop("overflow_to_sample_mapping")

    labels = []
    for chunk_idx, sample_idx in enumerate(sample_map):
        word_ids = tokenized_inputs.word_ids(batch_index=chunk_idx)
        label = examples["ner_tags"][sample_idx]
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Map special tokens to -100 to ignore them in loss
            if word_idx is None:
                label_ids.append(-100)
            # Assign label only to the first token of a split word
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs