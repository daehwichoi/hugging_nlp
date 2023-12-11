import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import iterative_train_test_split


def filter_labels(x):
    return [label_map[label] for label in x if label in label_map]

def balanced_split(df, test_size = 0.5):
    ind = 


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

    print(value_count)

    print(df_issues["labels"].head())

    df_issues["split"] = "unlabeled"
    mask = df_issues["labels"].apply(lambda x: len(x)) > 0
    df_issues[mask]["split"] = "labeled"
    df_issues["split"].value_counts().to_frame()

    print(df_issues)
    print(df_issues.columns)

    for column in ["title", "body", "labels"]:
        print(f"{column} : {df_issues[column].iloc[26][:500]}\n")

    df_issues["text"] = df_issues.apply(lambda x: x["title"] + "\n\n" + x["body"], axis=1)

    print(f"중복 제거 전 개수 : {len(df_issues)}")
    df_issues = df_issues.drop_duplicates(subset="text")
    print(f"중복 제거 후 개수 : {len(df_issues)}")

    mlb = MultiLabelBinarizer()
    mlb.fit([all_labels])
    print(mlb.transform([["tokenization", "new model"], ["pytorch"]]))