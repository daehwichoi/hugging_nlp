from datasets import get_dataset_config_names
from datasets import load_dataset

import torch
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering

from transformers import pipeline

if __name__ == '__main__':
    domains = get_dataset_config_names("subjqa")
    dataset = load_dataset("subjqa", name="electronics")

    dfs = {split: dset.to_pandas() for split, dset in dataset.flatten().items()}

    model_ckpt = "deepset/minilm-uncased-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    example = dfs["train"].iloc[0][["question", "context"]]
    tokenizer_example = tokenizer(example["question"], example["context"], return_overflowing_tokens=True, max_lengh=100, stride=25)


    question = "How much music can this hold?"
    context = """An MP3 is about 1 MB/miunute, so about 6000 hours depending on file size"""
    inputs = tokenizer(question, context, return_tensors="pt")
    print(inputs)
    print(tokenizer.decode(inputs["input_ids"][0]))

    model = AutoModelForQuestionAnswering.from_pretrained(model_ckpt)

    with torch.no_grad():
        outputs = model(**inputs)

    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits) + 1
    answer_span = inputs["input_ids"][0][start_idx:end_idx]
    answer = tokenizer.decode(answer_span)

    pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)
    print(pipe(question="Why is there no data", context=context, topk=3))
    # print(outputs)

    # print(dataset["train"]["answers"][1])
