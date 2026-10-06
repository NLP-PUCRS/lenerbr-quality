# Code to build Sentence Deduped dataset on top of HF fetched version of LeNER-Br

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
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

def get_all_sentences_of_split(dataset:DatasetDict, split:str) -> dict:
    if split == "dev":
        split = "validation"
    ner = dataset[split].to_pandas()
    return {" ".join(list(k)):(v,i) for k,v,i in zip(ner["tokens"].to_list(), ner["ner_tags"].to_list(), ner["id"].to_list())}

def calculate_sim_matrix(docs1:dict, docs2:dict):
    docs1_to_sim = list(docs1.keys())
    docs2_to_sim = list(docs2.keys())

    # 1. Fit once on all documents
    vectorizer = CountVectorizer(binary=True, token_pattern=r"\S+")
    vectorizer.fit(docs1_to_sim + docs2_to_sim)

    X_train = vectorizer.transform(docs1_to_sim).toarray().astype(bool)
    X_test = vectorizer.transform(docs2_to_sim).toarray().astype(bool)

    # 2. Compute pairwise Jaccard distance (1 - similarity)
    jaccard_dist = pairwise_distances(X_train, X_test, metric='jaccard', n_jobs=14)
    sim_matrix = 1.0 - jaccard_dist

    max_sim = np.max(sim_matrix)

    return sim_matrix, max_sim, np.where(sim_matrix == max_sim)

def get_filtered_dataset(save:bool=False, path:str="data/lenerbr_sentence_deduped") -> DatasetDict:

    dataset, _ = load_datasets()

    train_sentences = get_all_sentences_of_split(dataset, "train")
    test_sentences = get_all_sentences_of_split(dataset, "test")
    dev_sentences = get_all_sentences_of_split(dataset, "dev")

    _, _, sentence_pos_train_test = calculate_sim_matrix(train_sentences, test_sentences)
    _, _, sentence_pos_train_dev = calculate_sim_matrix(train_sentences, dev_sentences)
    _, _, sentence_pos_test_dev = calculate_sim_matrix(test_sentences, dev_sentences)

    leaked_sentences = {
        "train":{},
        "test":{},
        "dev":{}
    }

    for train_pos, test_pos in zip(sentence_pos_train_test[0], sentence_pos_train_test[1]):
        leaked_sentences["train"][train_sentences[list(train_sentences.keys())[train_pos]][1]] = (train_sentences[list(train_sentences.keys())[train_pos]][0], list(train_sentences.keys())[train_pos])
        leaked_sentences["test"][test_sentences[list(test_sentences.keys())[test_pos]][1]] = (test_sentences[list(test_sentences.keys())[test_pos]][0], list(test_sentences.keys())[test_pos])

    for train_pos, dev_pos in zip(sentence_pos_train_dev[0], sentence_pos_train_dev[1]):
        leaked_sentences["train"][train_sentences[list(train_sentences.keys())[train_pos]][1]] = (train_sentences[list(train_sentences.keys())[train_pos]][0], list(train_sentences.keys())[train_pos])
        leaked_sentences["dev"][dev_sentences[list(dev_sentences.keys())[dev_pos]][1]] = (dev_sentences[list(dev_sentences.keys())[dev_pos]][0], list(dev_sentences.keys())[dev_pos])

    for test_pos, dev_pos in zip(sentence_pos_test_dev[0], sentence_pos_test_dev[1]):
        leaked_sentences["test"][test_sentences[list(test_sentences.keys())[test_pos]][1]] = (test_sentences[list(test_sentences.keys())[test_pos]][0], list(test_sentences.keys())[test_pos])
        leaked_sentences["dev"][dev_sentences[list(dev_sentences.keys())[dev_pos]][1]] = (dev_sentences[list(dev_sentences.keys())[dev_pos]][0], list(dev_sentences.keys())[dev_pos])

    dataset["train"] = dataset["train"].filter(lambda x: x["id"] not in leaked_sentences["train"])
    dataset["validation"] = dataset["validation"].filter(lambda x: x["id"] not in leaked_sentences["dev"])

    if save:
        dataset.save_to_disk(path)
        print("Dataset saved.")
    return dataset

if __name__ == "__main__":
    get_filtered_dataset(save=True)