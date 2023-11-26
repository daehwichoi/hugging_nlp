import torch
from datasets import get_dataset_config_names
from datasets import load_dataset
from datasets import DatasetDict

from collections import defaultdict
import pandas as pd

if __name__ == '__main__':
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
