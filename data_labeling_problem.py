import numpy as np
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.model_selection import iterative_train_test_split

from datasets import Dataset, DatasetDict

from collections import defaultdict
from transformers import pipeline, set_seed

import nlpaug.augmenter.word as naw
import matplotlib.pyplot as plt


def filter_labels(x):
    return [label_map[label] for label in x if label in label_map]


def balanced_split(df, test_size=0.5):
    ind = np.expand_dims(np.arange(len(df)), axis=1)
    print(df)
    labels = mlb.transform(df["labels"])
    ind_train, _, ind_test, _ = iterative_train_test_split(ind, labels, test_size)
    return df.iloc[ind_train[:, 0]], df.iloc[ind_test[:, 0]]


def prepare_labels(batch):
    batch["label_ids"] = mlb.transform(batch["labels"])
    return batch


def zero_shot_pipeline(example):
    output = pipe(example["text"], all_labels, multi_label=True)
    example["predicted_labels"] = output["labels"]
    example["scores"] = output["scores"]
    return example


def get_pred(example, threshold=None, topk=None):
    pred = []
    if threshold:
        for label, score in zip(example["predicted_labels"], example["scores"]):
            if score >= threshold:
                pred.append(label)
    elif topk:
        for i in range(topk):
            pred.append(example["predicted_labels"][i])
    else:
        raise ValueError("threshold 또는 topk로 지정해야합니다. ")
    return {"pred_label_ids": list(np.squeeze(mlb.transform([pred])))}


def get_clf_report(ds):
    y_true = np.array(ds["label_ids"])
    y_pred = np.array(ds["pred_label_ids"])
    return classification_report(y_true, y_pred, target_names=mlb.classes_, zero_division=0, output_dict=True)


def plot_metics(micro_scores, macro_scores, sample_sizes, current_model):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for run in micro_scores.keys():
        if run == current_model:
            ax0.plot(sample_sizes, micro_scores[run], label=run, linewidth=2)
            ax1.plot(sample_sizes, macro_scores[run], label=run, linewidth=2)
        else:
            ax0.plot(sample_sizes, micro_scores[run], label=run, linestyle='dashed')
            ax1.plot(sample_sizes, macro_scores[run], label=run, linestyle='dashed')

    ax0.set_title("Micro F1 scores")
    ax1.set_title("Macro F1 scores")
    ax0.set_ylabel("Test set F1 score")
    ax0.legend(loc="lower right")
    for ax in [ax0, ax1]:
        ax.set_xlabel("Number of training samples")
        ax.set_xscale("log")
        ax.set_xticks(sample_sizes)
        ax.set_xticklabels(sample_sizes)
        ax.minorticks_off()
    plt.tight_layout()
    plt.show()


def augment_text(batch, transformations_per_example=1):
    set_seed(3)
    aug = naw.ContextualWordEmbsAug(model_path="distilbert-base-uncased", device='cuda', action='substitute')

    text_aug, label_ids = [], []
    for text, labels in zip(batch["text"], batch["label_ids"]):
        text_aug += [text]
        label_ids += [labels]

        for _ in range(transformations_per_example):
            text_aug += aug.augment(text)
            label_ids += [labels]
    return {"text": text_aug, "label_ids": label_ids}


if __name__ == '__main__':
    # if False:
    dataset_url = "https://git.io/nlp-with-transformers"
    df_issues = pd.read_json(dataset_url, lines=True)
    print(f"데이터프레임 크기 : {df_issues.shape}")

    df_issues["labels"] = df_issues["labels"].apply(lambda x: [k["name"] for k in x])
    value_count = df_issues["labels"].apply(lambda x: len(x)).value_counts().to_frame().T

    df_counts = df_issues["labels"].explode().value_counts()

    label_map = {"Core: Tokenization": "tokenization",
                 "New model": "new model",
                 "Core: Modeling": "model training",
                 "TensorFlow": "tensorflow or tf",
                 "Pytorch": "pytorch",
                 "Examples": "examples",
                 "Documentation": "documentation"}

    df_issues["labels"] = df_issues["labels"].apply(filter_labels)
    all_labels = list(label_map.values())

    df_issues["split"] = "unlabeled"
    mask = df_issues["labels"].apply(lambda x: len(x)) > 0
    df_issues.loc[mask, "split"] = "labeled"
    df_issues["split"].value_counts().to_frame()

    for column in ["title", "body", "labels"]:
        print(f"{column} : {df_issues[column].iloc[26][:500]}\n")

    df_issues["text"] = df_issues.apply(lambda x: x["title"] + "\n\n" + x["body"], axis=1)

    print(f"중복 제거 전 개수 : {len(df_issues)}")
    df_issues = df_issues.drop_duplicates(subset="text")
    print(f"중복 제거 후 개수 : {len(df_issues)}")

    mlb = MultiLabelBinarizer()
    mlb.fit([all_labels])

    df_clean = df_issues[["text", "labels", "split"]].reset_index(drop=True).copy()
    df_unsup = df_clean.loc[df_clean["split"] == 'unlabeled', ["text", "labels"]]
    df_sup = df_clean.loc[df_clean["split"] == 'labeled', ["text", "labels"]]

    np.random.seed(0)
    df_train, df_tmp = balanced_split(df_sup, test_size=0.5)
    df_valid, df_test = balanced_split(df_tmp, test_size=0.5)

    ds = DatasetDict({'train': Dataset.from_pandas(df_train.reset_index(drop=True)),
                      'valid': Dataset.from_pandas(df_valid.reset_index(drop=True)),
                      'test': Dataset.from_pandas(df_test.reset_index(drop=True)),
                      'unsup': Dataset.from_pandas(df_unsup.reset_index(drop=True))})

    print(ds)

    np.random.seed(0)
    all_indices = np.expand_dims(list(range(len(ds["train"]))), axis=1)
    indices_pool = all_indices
    labels = mlb.transform(ds["train"]["labels"])
    train_samples = [8, 16, 32, 64, 128]
    train_slices, last_k = [], 0

    # Trainset을 소규모로 나눔

    for i, k in enumerate(train_samples):
        indices_pool, labels, new_slice, _ = iterative_train_test_split(indices_pool, labels,
                                                                        (k - last_k) / len(labels))
        lask_k = k
        if i == 0:
            train_slices.append(new_slice)
        else:
            train_slices.append(np.concatenate((train_slices[-1], new_slice)))
    train_slices.append(all_indices)
    train_samples.append(len(ds["train"]))
    train_slices = [np.squeeze(train_slice) for train_slice in train_slices]



    # Naive Bayes Model (Base model)
    ds = ds.map(prepare_labels, batched=True)

    macro_scores, micro_scores = defaultdict(list), defaultdict(list)

    for train_slice in train_slices:
        ds_train_sample = ds["train"].select(train_slice)  # 해당 개수만큼 select
        ds_train_sample = ds_train_sample.map(augment_text, batched=True,
                                              remove_columns=ds_train_sample.column_names).shuffle(seed=42)
        y_train = np.array(ds_train_sample["label_ids"])
        y_test = np.array(ds["test"]["label_ids"])

        count_vector = CountVectorizer()
        x_train_counts = count_vector.fit_transform(ds_train_sample["text"])
        x_test_counts = count_vector.transform(ds["test"]["text"])

        classifier = BinaryRelevance(classifier=MultinomialNB())
        classifier.fit(x_train_counts, y_train)

        y_pred_test = classifier.predict(x_test_counts)
        clf_report = classification_report(y_test, y_pred_test, target_names=mlb.classes_, zero_division=0,
                                           output_dict=True)

        macro_scores["Naive Bayes"].append(clf_report["macro avg"]["f1-score"])
        micro_scores["Naive Bayes"].append(clf_report["micro avg"]["f1-score"])

    pipe = pipeline("zero-shot-classification", device=0)
    sample = ds["train"][0]
    print(f"레이블 : {sample['labels']}")
    output = pipe(sample["text"], all_labels, multi_label=True)
    print(output["sequence"][:400])

    for label, score in zip(output["labels"], output["scores"]):
        print(f"{label}, {score:.2f}")

    ds_zero_shot = ds["test"].map(zero_shot_pipeline)
    ds_zero_shot = ds_zero_shot.map(get_pred, fn_kwargs={'topk': 1})
    clf_report = get_clf_report(ds_zero_shot)
    for train_slice in train_slices:
        macro_scores["Zero Shot"].append(clf_report["macro avg"]["f1-score"])
        micro_scores["Zero Shot"].append(clf_report["micro avg"]["f1-score"])



    plot_metics(micro_scores, macro_scores, train_samples, 'Zero Shot')


#
# text = "Transformers are the most popular toys"
# print(f"원본 텍스트 : {text}")
# print(f"증식 텍스트 : {aug.augment(text)}")
