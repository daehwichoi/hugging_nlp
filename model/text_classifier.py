import torch
import torch.nn as nn

from model.transformer_encoder import TransformerEncoder


class TransformerForSequenceClassification(nn.Module):
    def __init__(self, config, num_labels=6):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, x):
        x = self.encoder(x)[:, 0, :]
        x = self.dropout(x)
        x = self.classifier(x)
        return x
