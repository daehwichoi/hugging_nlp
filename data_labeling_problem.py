import numpy as np
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split

from datasets import Dataset, DatasetDict

from collections import defaultdict
from transformers import pipeline


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
    output = pipe(example)


if __name__ == '__main__':
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

    print(macro_scores["Naive Bayes"])
    print(micro_scores["Naive Bayes"])

    # Bert를 이용한 zero-shot
    # pipe = pipeline("fill-mask", model = "bert-base-uncased")
    # movie_desc = "The main characters of the movie madacascar are a lion, a zebra, a giraffe, and a hippo."
    # prompt = "The movie is about [MASK]."
    #
    # output = pipe(movie_desc+prompt)
    #
    # for ele in output:
    #     print(f"토큰 {ele['token_str']}:\t {ele['score']:.3f}%")

    pipe = pipeline("zero-shot-classification", device=0)
    sample = ds["train"][0]
    print(f"레이블 : {sample['labels']}")
    output = pipe(sample["text"], all_labels, multi_label=True)
    print(output["sequence"][:400])

    for label, score in zip(output["labels"], output["scores"]):
        print(f"{label}, {score:.2f}")
