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
from collections import Counter
from collections import defaultdict


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

        num_feature = config.hidden_size // 2
        print(f"num feature : {num_feature}")
        self.embedding = nn.Embedding(config.vocab_size, num_feature)
        self.encoder_layer = nn.TransformerEncoderLayer(num_feature, 4)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, 1)
        self.lstm = nn.LSTM(num_feature, num_feature)


        self.bn1 = nn.BatchNorm1d(num_feature)
        self.fc1 = nn.Linear(num_feature, 128)
        self.dropout = nn.Dropout(0.5)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 1)

        self.initialize_layers()

    def initialize_layers(self):
        for layer in [self.bn1, self.bn2]:
            if isinstance(layer, nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)[:, 0, :]
        x = F.relu(self.fc1(self.bn1(x)))
        x = self.dropout(x)
        x = self.fc2(self.bn2(x))
        x = torch.sigmoid(x[:, 0])
        return x


def model_test():
    global idx, input, label, output
    model.eval()
    with torch.no_grad():
        total = 0
        right = 0
        for idx, (input, label) in enumerate(testloader):
            input = input.to(device)
            # label = torch.FloatTensor(label).to(device)
            label = torch.tensor(label, dtype=torch.float32).to(device)

            korean = tokenizer.decode(input[0,:].cpu())
            print(korean)


            output = model(input)
            print(output)
            # pred = torch.argmax(output, dim=-1)
            pred = (output > 0.5).float()
            # print(pred)

            total += input.shape[0]
            right += (pred == label).sum().item()
    print(f"acc : {round(100 * right / total, 3)} %")
    weight = model.state_dict()
    torch.save(weight, "./text_classify.pt")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # dataset load
    dataset = load_dataset("jeanlee/kmhas_korean_hate_speech")

    # pre_ckpt = "beomi/korean-hatespeech-classifier"
    pre_ckpt = "Tolerblanc/klue-bert-finetuned"
    tokenizer = AutoTokenizer.from_pretrained(pre_ckpt)
    config = AutoConfig.from_pretrained(pre_ckpt)

    train_encode_text = tokenizer(dataset["train"]["text"], padding="max_length", max_length= 300)
    test_encode_text = tokenizer(dataset["test"]["text"], padding="max_length", max_length= 300)

    class_label = {0: "origin", 1: "physical", 2: "politics", 3: "profanity", 4: "age", 5: "gender", 6: "race",
                   7: "religion", 8: "not_hate_speech"}
    class_label_num = {v: k for k, v in class_label.items()}

    # length : 680
    datas = []
    labels = []

    data_counter = defaultdict(int)
    for x, y in zip(train_encode_text["input_ids"], dataset["train"]["label"]):
        # if data_counter[y[0]] > 6000:
        #     continue

        # Label 0: 보통 나머지: 나쁜 말들
        datas.append(x)
        if y[0] == 8:
            labels.append(0)
        else:
            labels.append(1)
        # data_counter[y[0]] += 1

        # for i in range(len(y)):
        #     if data_counter[y[i]] > 6000:
        #         continue
        #     datas.append(x)
        #     labels.append(y[i])
        #     data_counter[y[i]] += 1
    data_counter = Counter(labels)
    print(data_counter)

    trainset = CustomDataset(np.array(datas), labels)

    datas = []
    labels = []

    for x, y in zip(test_encode_text["input_ids"], dataset["test"]["label"]):
        datas.append(x)
        # labels.append(y[0])
        if y[0] == 8:
            labels.append(0)
        else:
            labels.append(1)

    testset = CustomDataset(np.array(datas), labels)

    train_batch_size = 256
    test_batch_size = 512
    trainloader = DataLoader(trainset, shuffle=True, batch_size=train_batch_size, drop_last=True)
    testloader = DataLoader(testset, shuffle=False, batch_size=test_batch_size, drop_last=True)
    model = CustomModel(config).to(device)

    criterion = nn.BCELoss()
    optim = torch.optim.Adam(model.parameters(), lr=2e-5)

    n_epochs = 10

    model.train()
    for epoch in range(n_epochs):
        for idx, (input, label) in enumerate(trainloader):
            input = input.to(device)
            # label = torch.FloatTensor(label).to(device)
            label = torch.tensor(label, dtype=torch.float32).to(device)

            optim.zero_grad()
            output = model(input)
            # print(output)
            # print(torch.argmax(output, dim=-1))
            loss = criterion(output, label)
            loss.backward()
            optim.step()

            if idx % 100 == 0:
                print(f"{idx}'th Loss : {loss / (100 * train_batch_size)}")
        model_test()

    load_pretrained_model = False
    if load_pretrained_model:
        model.load_state_dict(torch.load("text_classify.pt"))

    model_test()
    # train_loader = DataLoader()

    # print(train_encode_text["input_ids"][0])

    # model 구성
    # model = nn.Sequential([])
