from datasets import load_dataset
from datasets import load_metric

from transformers import pipeline

import torch
import numpy as np

from pathlib import Path
import time


class PerformanceBenchmark:
    def __init__(self, pipeline, dataset, optim_type="BERT baseline"):
        self.pipeline = pipeline
        self.dataset = dataset
        self.optim_type = optim_type

    def compute_accuracy(self):
        accuracy_score = load_metric("accuracy")
        preds, labels = [], []
        for example in self.dataset:
            pred = self.pipeline(example["text"])[0]["label"]
            label = example["intent"]
            preds.append(intent.str2int(pred))
            labels.append(label)
        accuracy = accuracy_score.compute(predictions=preds, references=labels)
        print("테스트 정확도 {0}".format(float(accuracy["accuracy"])))
        return accuracy

    def compute_size(self):
        state_dict = self.pipeline.model.state_dict()
        tmp_path = Path("./model.pt")
        torch.save(state_dict, tmp_path)
        size_mb = Path(tmp_path).stat().st_size / (1024 * 1024)
        tmp_path.unlink()
        print(f"모델 크기(MB) : {size_mb}")
        return {"size_mb": size_mb}

    def time_pipeline(self,
                      query="Hey, I'd like to rent a vehicle from Nov 1st to Nov 15th in Paris and I need a 15 passenger van",
                      latencies=[]):

        for _ in range(10):
            _ = self.pipeline(query)

        for _ in range(100):
            start_time = time.perf_counter()
        _ = self.pipeline(query)
        latency = time.perf_counter() - start_time
        latencies.append(latency)

        time_avg_ms = 1000 * np.mean(latencies)
        time_std_ms = 1000 * np.std(latencies)

        print(f"평균 latency : {time_avg_ms} +- {time_std_ms} ms")
        return {"time_avg_ms": time_avg_ms, "time_std_ms": time_std_ms}

    def run_benchmark(self):
        metrics = {}
        metrics[self.optim_type] = self.compute_size()
        metrics[self.optim_type].update(self.time_pipeline())
        metrics[self.optim_type].update(self.compute_accuracy())
        return metrics


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    bert_ckpt = "transformersbook/bert-base-uncased-finetuned-clinc"
    pipe = pipeline("text-classification", model=bert_ckpt, device=device)

    clinc = load_dataset("clinc_oos", "plus")
    intent = clinc["test"].features["intent"]

    pb = PerformanceBenchmark(pipe, clinc["test"])
    perf_metrics = pb.run_benchmark()

    # data = pipe(query)
    # print(data)

    # print(list(pipe.model.state_dict().items()))

    # torch.save(pipe.model.state_dict(), "model.pt")
