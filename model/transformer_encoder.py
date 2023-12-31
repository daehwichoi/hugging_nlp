import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt


def scaled_dot_product_attention(query, key, value, mask=None):
    score = torch.bmm(query, key.transpose(1, 2)) / sqrt(key.size(-1))
    if mask is not None:
        score = score.masked_fill(mask == 0, float("-inf"))

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


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout()

    def forward(self, x):
        # Batch X sequence X hidden_size
        seq_length = x.size(1)
        positions_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
        token_embeddings = self.token_embeddings(x)
        position_embeddings = self.position_embeddings(positions_ids)

        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config.hidden_size, config.n_heads)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        hidden_state = self.layer_norm1(x)
        x = x + self.attention(hidden_state)
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return x
