import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split


def filter_labels(x):
    return [label_map[label] for label in x if label in label_map]


def balanced_split(df, test_size=0.5):
    ind = np.expand_dims(np.arange(len(df)), axis=1)
    print(df)
    labels = mlb.transform(df["labels"])
    ind_train, _, ind_test, _ = iterative_train_test_split(ind, labels, test_size)
    return df.iloc[ind_train[:, 0]], df.iloc[ind_test[:, 0]]


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
    df_issues.loc[mask,"split"] = "labeled"
    df_issues["split"].value_counts().to_frame()

    for column in ["title", "body", "labels"]:
        print(f"{column} : {df_issues[column].iloc[26][:500]}\n")

    df_issues["text"] = df_issues.apply(lambda x: x["title"] + "\n\n" + x["body"], axis=1)

    print(f"중복 제거 전 개수 : {len(df_issues)}")
    df_issues = df_issues.drop_duplicates(subset="text")
    print(f"중복 제거 후 개수 : {len(df_issues)}")

    mlb = MultiLabelBinarizer()
    mlb.fit([all_labels])
    print(mlb.transform([["tokenization", "new model"], ["pytorch"]]))

    df_clean = df_issues[["text", "labels", "split"]].reset_index(drop=True).copy()
    df_unsup = df_clean.loc[df_clean["split"] == 'unlabeled', ["text", "labels"]]
    df_sup = df_clean.loc[df_clean["split"] == 'labeled', ["text", "labels"]]
    print(df_sup)

    np.random.seed(0)
    df_train, df_tmp = balanced_split(df_sup, test_size=0.5)
    df_valid, df_test = balanced_split(df_tmp, test_size=0.5)
    print(df_train)
