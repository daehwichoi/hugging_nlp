import pandas as pd
from datasets import list_datasets, load_dataset

if __name__ == '__main__':
    # all_dataset = list_datasets()
    # print(len(all_dataset))
    # print(all_dataset[:10])

    dataset = load_dataset("emotion")
    print(dataset['train'][0])

    # train_data = dataset["train"]
    # test_data = dataset["test"]

    train_df = pd.DataFrame(dataset['train'])
    train_df['label_name'] = train_df["label"].apply(lambda x: dataset["train"].features["label"].int2str(x))
    print(train_df)