import torch
from datasets import get_dataset_config_names
from datasets import load_dataset
from datasets import DatasetDict

from collections import defaultdict
from collections import Counter
import pandas as pd
import torch

from transformers import AutoConfig
from transformers import AutoTokenizer

from model.token_classifier import XLMRobertaForTokenClassification


def create_tag_names(batch):
    return {"ner_tags_str": [tags.int2str(idx) for idx in batch["ner_tags"]]}


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    xtreme_subsets = get_dataset_config_names("xtreme")
    focus_data = [i for i in xtreme_subsets if i.startswith("PAN")]

    language = ['ko', 'en']
    fraction = [0.5, 0.5]

    panx_ch = defaultdict(DatasetDict)
    print(panx_ch)

    for lang, frac in zip(language, fraction):
        data = load_dataset("xtreme", name="PAN-X.ko")

        for split in data:
            panx_ch[lang][split] = (data[split].shuffle(seed=0).select(range(int(frac * data[split].num_rows))))

    data_len = pd.DataFrame({lang: [panx_ch[lang]["train"].num_rows] for lang in language}, index=["example_num"])
    print(data_len)

    element = panx_ch["ko"]["train"][0]
    for key, value in element.items():
        print(key, value)

    for key, value in panx_ch["ko"]["train"].features.items():
        print(key, value)

    tags = panx_ch["ko"]["train"].features["ner_tags"].feature

    panx_ko = panx_ch["ko"].map(create_tag_names)

    df = pd.DataFrame(panx_ko["train"])

    index2tag = {idx: tag for idx, tag in enumerate(tags.names)}
    tag2index = {tag: idx for idx, tag in enumerate(tags.names)}

    text = "I love you, Korean!"
    xlmr_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    xlmr_config = AutoConfig.from_pretrained("xlm-roberta-base", num_labels=tags.num_classes, id2label=index2tag,
                                             label2id=tag2index)

    input_ids = xlmr_tokenizer.encode(text, return_tensors="pt")
    xlmr_model = XLMRobertaForTokenClassification.from_pretrained("xlm-roberta-base", config=xlmr_config).to(device)

    outputs = xlmr_model(input_ids.to(device)).logits
    predictions = torch.argmax(outputs, dim=-1)

    print(input_ids)

    # print(xlmr_config)

    # print(panx_ko["train"]["ner_tags_str"])
    # print(df)
