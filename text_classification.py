import pandas as pd
from datasets import list_datasets, load_dataset
from transformers import AutoTokenizer, AutoModel
import torch

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
    encode_text = tokenizer(text, return_tensors= "pt")
    # token = tokenizer.convert_ids_to_tokens(encode_text.input_ids)

    def tokenize(batch):
        return tokenizer(batch['text'], padding=True, truncation=True)

    print(tokenize(dataset['train'][:2]))

    # emotion_encoded = dataset.map(tokenize, batched=True, batch_size=None)
    # print(emotion_encoded)
    #
    device= 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModel.from_pretrained('distilbert-base-uncased').to(device)


    inputs = {k:v.to(device) for k,v in encode_text.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    print(outputs)
    print(outputs.shape)



    # print(encode_text)
    # print(token)



