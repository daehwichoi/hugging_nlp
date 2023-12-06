import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from datasets import load_dataset
import numpy as np

import random


class CustomDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.LongTensor(self.data[idx]), self.label[idx]


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding()
        self.encoder_layer = nn.TransformerEncoderLayer(680, 4)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, 4)
        self.fc = nn.Linear(100, 2)

    def forward(self, x):
        x = self.transformer(x)
        print(x.shape)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # dataset load
    dataset = load_dataset("jeanlee/kmhas_korean_hate_speech")

    tokenizer = AutoTokenizer.from_pretrained("quantumaikr/llama-2-70b-fb16-korean")
    train_encode_text = tokenizer(dataset["train"]["text"], padding=True)
    # length : 680
    datas = []
    labels = []

    for x,y in zip(train_encode_text["input_ids"], dataset["train"]["label"]):
        for i in range(len(y)):
            datas.append(x)
            labels.append(y[i])

    datas = np.array(datas)

    trainset = CustomDataset(datas, labels)
    trainloader = DataLoader(trainset, shuffle=True, batch_size=128, drop_last=True)
    model = CustomModel().to(device)

    for idx, (input, label) in enumerate(trainloader):
        input = input.to(device)
        output = model(input)
        print(output)


    # train_loader = DataLoader()

    # print(train_encode_text["input_ids"][0])

    # model 구성
    # model = nn.Sequential([])
