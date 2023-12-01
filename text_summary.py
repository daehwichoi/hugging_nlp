import pandas as pd
from datasets import load_dataset
from datasets import load_metric

import nltk
from nltk.tokenize import sent_tokenize

from transformers import pipeline, set_seed
import numpy as np



def base_summary_model(text):
    return "\n".join(sent_tokenize(text)[:3])


if __name__ == '__main__':
    dataset = load_dataset("cnn_dailymail", version="3.0.0")
    print(dataset["train"])

    sample_text = dataset["train"][1]["article"][:2000]
    summaries = {}

    string = f"In this week’s patch we’re focusing on addressing a few outliers and refining changes made in a few of the last patches, like those to K’Sante. We’re focusing more and more on getting the 2024 changes polished and ready for 14.1, so be aware that this patch and 13.24 may be a bit smaller and less complex than usual as 14.1 will undoubtedly shake things up! In other news, we have some un-bee-lievably adorable Bee skins coming out this patch, a round of Nexus Blitz and ARAM balance adjustments, and last but certainly not least, this is the patch the Riot ID changes go into effect. We’ve made a few changes so make sure you check those out down below. TFT has a new set launching this patch that has you mix your own track by combining different bands (traits) on your board! Read all about The Remix Rumble in the TFT patch notes here! "
    nltk.download("punkt")

    set_seed(42)
    pipe = pipeline("summarization", model='facebook/bart-large-cnn')
    pipe_out = pipe(sample_text)
    summaries["bart"] = '\n'.join(sent_tokenize(pipe_out[0]["summary_text"]))

    # pipe = pipeline("text-generation", model="gpt2-xl")
    # gpt2_query = sample_text + "\nTL;DR:\n"
    # pipe_out = pipe(gpt2_query, max_length=512, clean_up_tokenization_spaces=True)
    # summaries["gpt2"] = "\n".join(sent_tokenize(pipe_out[0]["generated_text"][len(gpt2_query):]))
    summaries["base"] = base_summary_model(sample_text)
    print(summaries)

    bleu_metric = load_metric("sacrebleu")
    bleu_metric.add(prediction=summaries["bart"], reference = [sample_text])
    results = bleu_metric.compute(smooth_method = 'floor', smooth_value=0)
    results["precisions"] = [np.round(p,2) for p in results["precisions"]]
    print(pd.DataFrame(results))