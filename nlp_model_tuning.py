from datasets import load_dataset
from datasets import load_metric

from transformers import pipeline
import torch
from pathlib import Path

import time

clinc = load_dataset("clinc_oos", "plus")
intent = clinc["test"].features["intent"]


class PerformanceBenchmark:
    def __init__(self, pipeline, dataset, optim_type="BERT baseline"):
        self.pipeline = pipeline
        self.datset = dataset
        self.optim_type = optim_type

    def compute_accuracy(self):
        preds, labels = [], []
        for example in self.dataset:
            pred = self.pipeline(example["text"])[0]["label"]
            label = example["intent"]
            preds.append(intent.str2int(pred))
            labels.append(label)
        accuracy = accuracy_score.compute(predictions=preds, references=labels)
        print(f"테스트 정확도 {0}".format(accuracy["accuracy"]))
        return accuracy

    def compute_size(self):
        state_dict = self.pipeline.model.state_dict()
        tmp_path = Path("./model.pt")
        torch.save(state_dict, tmp_path)
        size_mb = Path(tmp_path).stat().st_size() / (1024*1024)
        tmp_path.unlink()
        print(f"모델 크기(MB) : {size_mb}")
        return {"size_mb": size_mb}

    def time_pipeline(self):
        pass

    def run_benchmark(self):
        metrics = {}
        metrics[self.optim_type] = self.compute_size()
        metrics[self.optim_type].update(self.time_pipeline())
        metrics[self.optim_type].update(self.compute_accuracy())
        return metrics


if __name__ == "__main__":
    bert_ckpt = "transformersbook/bert-base-uncased-finetuned-clinc"
    pipe = pipeline("text-classification", model=bert_ckpt)

    accuracy_score = load_metric("accuracy")
    query = "Hey, I'd like to rent a vehicle from Nov 1st to Nov 15th in Paris and I need a 15 passenger van"
    # data = pipe(query)
    # print(data)
    for _ in range(3):
        start_time = time.perf_counter()
        _ = pipe(query)
        latnecy = time.perf_counter()-start_time
        print(latnecy)

    # print(list(pipe.model.state_dict().items()))

    # torch.save(pipe.model.state_dict(), "model.pt")
