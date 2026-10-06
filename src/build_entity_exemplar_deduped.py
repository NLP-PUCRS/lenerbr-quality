# Code to build Entity Exemplar Deduped dataset on top of the dataset created by src/build_sentence_deduped.py

import argparse
import csv
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk

Entity = tuple[str, tuple[str, ...]]  # (entity_class, token_span)


def extract_entities(tokens: list[str], tags: list[str]) -> set[Entity]:
    """Extracts (entity_class, token_span) exemplars from a BIO-tagged sentence, excluding O."""
    entities: set[Entity] = set()
    span: list[str] = []
    span_class: str | None = None

    def flush():
        if span:
            entities.add((span_class, tuple(span)))

    for token, tag in zip(tokens, tags):
        if tag == "O":
            flush()
            span, span_class = [], None
            continue

        prefix, entity_class = tag.split("-", 1)
        if prefix == "B" or entity_class != span_class:
            flush()
            span, span_class = [token], entity_class
        else:  # prefix == "I" continuing a span of the same class
            span.append(token)

    flush()
    return entities


def collect_split_entities(dataset: Dataset, label_names: list[str]) -> set[Entity]:
    """Union of all entity exemplars across every sentence in a split."""
    all_entities: set[Entity] = set()
    for example in dataset:
        tags = [label_names[t] for t in example["ner_tags"]]
        all_entities.update(extract_entities(example["tokens"], tags))
    return all_entities


def filter_split(
    dataset: Dataset,
    label_names: list[str],
    split_name: str,
    reference_entity_sources: dict[str, set[Entity]],
) -> tuple[Dataset, list[dict]]:
    """Drops sentences that contain any entity exemplar present in reference_entity_sources.

    Returns the filtered dataset plus a log with one row per (removed sentence,
    matching entity) pair, naming the reference split(s) each match came from.
    """
    reference_entities: set[Entity] = set()
    for entities in reference_entity_sources.values():
        reference_entities.update(entities)

    keep_indices = []
    removal_log = []
    for i, example in enumerate(dataset):
        tags = [label_names[t] for t in example["ner_tags"]]
        sentence_entities = extract_entities(example["tokens"], tags)
        matches = sentence_entities.intersection(reference_entities)

        if not matches:
            keep_indices.append(i)
            continue

        for entity_class, span in sorted(matches):
            matched_in = [
                source for source, entities in reference_entity_sources.items()
                if (entity_class, span) in entities
            ]
            removal_log.append({
                "removed_from_split": split_name,
                "sentence_id": example.get("id"),
                "sentence": " ".join(example["tokens"]),
                "entity_class": entity_class,
                "matched_token_span": " ".join(span),
                "matched_in_splits": "+".join(matched_in),
            })

    return dataset.select(keep_indices), removal_log


def build_entity_filtered_dataset(raw: DatasetDict) -> tuple[DatasetDict, list[dict]]:
    label_names = raw["train"].features["ner_tags"].feature.names

    val_entities = collect_split_entities(raw["validation"], label_names)
    test_entities = collect_split_entities(raw["test"], label_names)

    train_filtered, train_log = filter_split(
        raw["train"], label_names, "train",
        {"validation": val_entities, "test": test_entities},
    )
    validation_filtered, validation_log = filter_split(
        raw["validation"], label_names, "validation",
        {"test": test_entities},
    )

    filtered = DatasetDict({
        "train": train_filtered,
        "validation": validation_filtered,
        "test": raw["test"],
    })
    return filtered, train_log + validation_log


def write_removal_log(removal_log: list[dict], path: str):
    fieldnames = [
        "removed_from_split", "sentence_id", "sentence",
        "entity_class", "matched_token_span", "matched_in_splits",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(removal_log)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/lenerbr_sentence_deduped")
    parser.add_argument("--output", default="data/lenerbr_entity_deduped")
    parser.add_argument("--log", default="results/entity_filtered_removals.csv")
    args = parser.parse_args()

    raw = load_from_disk(args.source)
    filtered, removal_log = build_entity_filtered_dataset(raw)

    for split in ("train", "validation", "test"):
        before, after = len(raw[split]), len(filtered[split])
        print(f"{split}: {before} -> {after} ({before - after} removed)")

    write_removal_log(removal_log, args.log)
    removed_sentences = {(row["removed_from_split"], row["sentence_id"]) for row in removal_log}
    print(f"Logged {len(removal_log)} matches for {len(removed_sentences)} removed sentences to {args.log}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    filtered.save_to_disk(args.output)
    print(f"Saved filtered dataset to {args.output}")


if __name__ == "__main__":
    main()
