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
    def __init__(self, config):
        super().__init__()

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size // 4)
        self.encoder_layer = nn.TransformerEncoderLayer(config.hidden_size // 4, 4)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, 4)
        self.fc = nn.Linear(config.hidden_size // 4, 9)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)[:, 0, :]
        x = self.fc(x)
        return x


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # dataset load
    dataset = load_dataset("jeanlee/kmhas_korean_hate_speech")

    pre_ckpt = "quantumaikr/llama-2-70b-fb16-korean"
    tokenizer = AutoTokenizer.from_pretrained(pre_ckpt)
    config = AutoConfig.from_pretrained(pre_ckpt)

    train_encode_text = tokenizer(dataset["train"]["text"], padding=True)
    test_encode_text = tokenizer(dataset["test"]["text"], padding=True)
    # length : 680
    datas = []
    labels = []

    for x, y in zip(train_encode_text["input_ids"], dataset["train"]["label"]):
        for i in range(len(y)):
            datas.append(x)
            labels.append(y[i])
    datas = np.array(datas)
    trainset = CustomDataset(datas, labels)

    datas = []
    labels = []

    for x, y in zip(test_encode_text["input_ids"], dataset["test"]["label"]):
        for i in range(len(y)):
            datas.append(x)
            labels.append(y[i])

    datas = np.array(datas)
    testset = CustomDataset(datas, labels)

    trainloader = DataLoader(trainset, shuffle=True, batch_size=16, drop_last=True)
    testloader = DataLoader(testset, shuffle=False, batch_size=1024, drop_last=False)
    model = CustomModel(config).to(device)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.002)

    model.train()
    for idx, (input, label) in enumerate(trainloader):
        input = input.to(device)
        label = torch.LongTensor(label).to(device)

        optim.zero_grad()
        output = model(input)
        loss = criterion(output, label)
        loss.backward()
        optim.step()

    model.eval()
    with torch.no_grad():
        for idx, (input, label) in enumerate(testset):
            input = input.to(device)
            label = torch.LongTensor(label).to(device)

            output = model(input)
            pred = torch.argmax(output, dim=-1)
            print(pred)

    # train_loader = DataLoader()

    # print(train_encode_text["input_ids"][0])

    # model 구성
    # model = nn.Sequential([])
