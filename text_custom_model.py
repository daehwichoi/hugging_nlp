import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from datasets import load_dataset

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # dataset load
    dataset = load_dataset("jeanlee/kmhas_korean_hate_speech")

    tokenizer = AutoTokenizer.from_pretrained("quantumaikr/llama-2-70b-fb16-korean")
    train_encode_text = tokenizer(dataset["train"]["text"])
    print(train_encode_text.keys())




    # model 구성
    # model = nn.Sequential([])


