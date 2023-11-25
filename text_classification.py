import pandas as pd
from datasets import list_datasets, load_dataset
from transformers import AutoTokenizer, AutoModel

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression

from collections import Counter

if __name__ == '__main__':
    all_dataset = list_datasets()
    print(len(all_dataset))
    print(all_dataset[:10])

    dataset = load_dataset("emotion")
    print(dataset['train'][0])

    # train_data = dataset["train"]
    # test_data = dataset["test"]

    # train_df = pd.DataFrame(dataset['train'])
    # test_df = pd.DataFrame(dataset['test'])
    # train_df['label_name'] = train_df["label"].apply(lambda x: dataset["train"].features["label"].int2str(x))
    # test_df['label_name'] = train_df["label"].apply(lambda x: dataset["test"].features["label"].int2str(x))

    # Class 분포 확인
    # data = Counter(train_df['label_name'])
    # print("Train class 분포 : ", dict(data))
    # data = Counter(test_df['label_name'])
    # print("Test class 분포 : ", dict(data))

    # Token 개수 확인
    # train_df["num_of_token"] = train_df["text"].apply(lambda x : len(x.split(' ')))
    # test_df["num_of_token"] = test_df["text"].apply(lambda x : len(x.split(' ')))
    # print(train_df)
    # print(test_df)

    # print("Train 최대 token", max(train_df["num_of_token"]))
    # print("Test 최대 token", max(test_df["num_of_token"]))

    # Tokenizer 이용
    model_ckpt = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    print("token 숫자", tokenizer.vocab_size)

    text = "I love you daehwi"
    encode_text = tokenizer(text, return_tensors="pt")


    # token = tokenizer.convert_ids_to_tokens(encode_text.input_ids)

    def tokenize(batch):
        return tokenizer(batch['text'], padding=True, truncation=True)


    # print(tokenize(dataset['train'][:2]))

    emotion_encoded = dataset.map(tokenize, batched=True, batch_size=None)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModel.from_pretrained('distilbert-base-uncased').to(device)
    print(tokenizer.model_input_names)

    def extract_hidden_state(batch):
        inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
        print(inputs)
        with torch.no_grad():
            outputs = model(**inputs).last_hidden_state
        return {"hidden_state": outputs[:, 0].cpu().numpy()}


    emotion_encoded.set_format('torch', columns=["input_ids", "attention_mask", "label"])
    emotions_hidden = emotion_encoded.map(extract_hidden_state, batched=True)
    print(emotions_hidden)

    emotion_encoded.set_format('torch', columns=["input_ids", "attention_mask", "label"])
    emotions_hidden = emotion_encoded.map(extract_hidden_state, batched=True)

    X_train = np.array(emotions_hidden['train']['hidden_state'])
    X_valid = np.array(emotions_hidden['validation']['hidden_state'])
    Y_train = np.array(emotions_hidden['train']['label'])
    Y_valid = np.array(emotions_hidden['validation']['label'])

    lr_clf = LogisticRegression(max_iter=3000)
    lr_clf.fit(X_train, Y_train)
    print(lr_clf.score(X_valid, Y_valid))