from datasets import load_dataset

from transformers import pipeline


class PerformanceBenchmark:
    def __init__(self, pipeline, dataset, optim_type="BERT baseline"):
        self.pipeline = pipeline
        self.datset = dataset
        self.optim_type = optim_type

    def compute_accuracy(self):
        pass

    def compute_size(self):
        pass

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

    # query = "Hey, I'd like to rent a vehicle from Nov 1st to Nov 15th in Paris and I need a 15 passenger van"
    # data = pipe(query)
    # print(data)

    # 데이터 분석
    clinc = load_dataset("clinc_oos", "plus")
    print(clinc)
    # print(clinc["test"][0])
    sample = clinc["test"][0]
    intent = clinc["test"].features["intent"]
    print(intent)
    print(intent.int2str(sample["intent"]))
