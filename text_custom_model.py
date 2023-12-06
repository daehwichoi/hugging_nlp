import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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
        if len(self.label[idx]) > 1:
            # print(random.randint(len(self.label[idx])))
            label = self.label[idx][0]
            print(label)
        else:
            label = self.label[idx]

        return torch.LongTensor(self.data[idx]).unsqueeze(0), torch.LongTensor(label).unsqueeze(0)


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(4, 4)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, 4)
        self.fc = nn.Linear(100, 2)

    def forward(self, x):
        x = self.transformer(x)
        x = self.fc(x)
        print(x)
        return x


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # dataset load
    dataset = load_dataset("jeanlee/kmhas_korean_hate_speech")

    tokenizer = AutoTokenizer.from_pretrained("quantumaikr/llama-2-70b-fb16-korean")
    train_encode_text = tokenizer(dataset["train"]["text"], padding=True)
    # length : 680
    # print(train_encode_text["input_ids"])
    train_encode_text = np.array(train_encode_text["input_ids"])
    train_labels = np.array(dataset["train"]["label"])

    trainset = CustomDataset(train_encode_text, train_labels)
    trainloader = DataLoader(trainset, shuffle=True, batch_size=128)

    for idx, (input, label) in enumerate(trainloader):
        print(input, label)

    model = CustomModel()

    # train_loader = DataLoader()

    # print(train_encode_text["input_ids"][0])

    # model 구성
    # model = nn.Sequential([])
