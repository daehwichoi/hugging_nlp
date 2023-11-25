import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig
from transformers import AutoTokenizer

from math import sqrt


def scaled_dot_product_attention(query, key, value):
    score = torch.bmm(query, key.transpose(1, 2)) / sqrt(key.size(-1))
    weights = F.softmax(score, dim=-1)
    return torch.bmm(weights, value)


class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_state):
        attention_output = scaled_dot_product_attention(self.q(hidden_state), self.k(hidden_state),
                                                        self.v(hidden_state))
        return attention_output


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([AttentionHead(embed_dim, head_dim) for _ in range(num_heads)])
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state):
        x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
        x = self.fc(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.seq_classif_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


if __name__ == '__main__':
    text = "I love korea"

    model_ckpt = "distilbert-base-uncased"
    config = AutoConfig.from_pretrained(model_ckpt)
    print(config)

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, return_tensors="pt", add_special_tokens=False)
    embed = tokenizer(text)
    print(embed.input_ids)

    token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
    text_embed = token_emb(torch.LongTensor(embed.input_ids)).unsqueeze(0)
    print(text_embed.shape)

    # 유사도 계산
    # query = text_embed
    # key = text_embed
    # value = text_embed
    # attention_output = scaled_dot_product_attention(query, key, value)
    # print(attention_output)

    # score = torch.bmm(query, key.transpose(1, 2)) / sqrt(key.size(-1))
    # weights = F.softmax(score, dim=-1)
    # att_output = torch.bmm(weights, value)

    # Single Head 적용
    # model = AttentionHead(text_embed.size(-1), 100)
    # output = model(text_embed)
    # print(output)

    # Multi Head 적용
    model = MultiHeadAttention(config.hidden_size, config.num_attention_heads)
    output = model(text_embed)

    feed_forward = FeedForward(config)
    f_output = feed_forward(output)
    print(f_output)
