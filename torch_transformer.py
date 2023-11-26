import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig
from transformers import AutoTokenizer

from model.classification_model import TransformerForSequenceClassification

if __name__ == '__main__':
    text = "I love korea"

    model_ckpt = "distilbert-base-uncased"
    config = AutoConfig.from_pretrained(model_ckpt)
    print(config)

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, return_tensors="pt", add_special_tokens=False)
    embed = tokenizer(text)
    embed.input_ids = torch.LongTensor(embed.input_ids).unsqueeze(0)

    print(embed.input_ids)

    # token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
    # text_embed = token_emb(torch.LongTensor(embed.input_ids)).unsqueeze(0)
    # print(text_embed.shape)

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

    # Encoder 단
    # token_embed = torch.LongTensor(embed.input_ids).unsqueeze(0)
    # model = TransformerForSequenceClassification(config, num_labels=2)
    # output = model(token_embed)

    # Decoder 단
    seq_len = embed.input_ids.size(-1)
    # decoder 단에서는 앞 부분을 볼 수 없기 때문에 mask 필요
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)



    print(mask[0])