import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoTokenizer
from datasets import load_dataset

if __name__ == '__main__':
    tokenizer_model = "sepidmnorozy/Korean_sentiment"
    dataset = load_dataset("sepidmnorozy/Korean_sentiment")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)



    pass
