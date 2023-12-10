import pandas as pd


if __name__ == '__main__':
    dataset_url = "https://git.io/nlp-with-transformers"
    df_issues = pd.read_json(dataset_url, lines=True)
    print(f"데이터프레임 크기 : {df_issues.shape}")

    df_issues["labels"] = df_issues["labels"].apply(lambda x: [k["name"] for k in x])
    value_count = df_issues["labels"].apply(lambda x: len(x)).value_counts().to_frame().T

    df_counts = df_issues["labels"].explode().value_counts()


    print(value_count)

    print(df_issues["labels"].head())