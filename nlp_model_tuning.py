from datasets import load_dataset
from datasets import load_metric

from transformers import pipeline
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pathlib import Path
import time
import optuna


def tokenize_text(batch):
    return student_tokenizer(batch["text"], truncation=True)


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs_stu = model(**inputs)
        loss_ce = outputs_stu.loss
        logits_stu = outputs_stu.logits

        with torch.no_grad():
            outputs_tea = self.teacher_model(**inputs)
            logits_tea = outputs_tea.logits
        loss_fct = nn.KLDivLoss(reduction="batchmean")
        loss_hd = self.args.temperature ** 2 * loss_fct(F.log_softmax(logits_stu / self.args.temperature, dim=-1),
                                                        F.softmax(logits_tea / self.args.temperature, dim=-1))

        loss = self.args.alpha * loss_ce + (1. - self.args.alpha) * loss_hd
        return (loss, outputs_stu) if return_outputs else loss


class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature


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


def compute_metrics(pred):
    accuracy_score = load_metric("accuracy")
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_score.compute(predictions=predictions, references=labels)


def hp_space(trial):
    return {"num_train_epochs": trial.suggest_int("num_train_epochs", 5, 10),
            "alpha": trial.suggest_float("alpha", 0, 1), "temperature": trial.suggest_int("temperature", 2, 20)}


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    bert_ckpt = "transformersbook/bert-base-uncased-finetuned-clinc"
    pipe = pipeline("text-classification", model=bert_ckpt, device=device)

    clinc = load_dataset("clinc_oos", "plus")
    intent = clinc["test"].features["intent"]

    # pb = PerformanceBenchmark(pipe, clinc["test"])
    # perf_metrics = pb.run_benchmark()

    student_ckpt = "distilbert-base-uncased"
    student_tokenizer = AutoTokenizer.from_pretrained(student_ckpt)
    clinc_enc = clinc.map(tokenize_text, batched=True, remove_columns=["text"])
    clinc_enc = clinc_enc.rename_column("intent", "labels")

    batch_size = 48
    finetuned_ckpt = "distilbert-base-uncased-finetuned-clinc"

    student_training_args = DistillationTrainingArguments(output_dir=finetuned_ckpt, evaluation_strategy="epoch",
                                                          num_train_epochs=5, learning_rate=2e-5,
                                                          per_device_train_batch_size=batch_size,
                                                          per_device_eval_batch_size=batch_size, alpha=1,
                                                          weight_decay=0.01, push_to_hub=True)

    id2label = pipe.model.config.id2label
    label2id = pipe.model.config.label2id

    num_labels = intent.num_classes
    student_config = (
        AutoConfig.from_pretrained(student_ckpt, num_labels=num_labels, id2label=id2label, label2id=label2id))


    def student_init(): return (
        AutoModelForSequenceClassification.from_pretrained(student_ckpt, config=student_config).to(device))


    teacher_ckpt = "transformersbook/bert-base-uncased-finetuned-clinc"
    teacher_model = (AutoModelForSequenceClassification.from_pretrained(teacher_ckpt, num_labels=num_labels).to(device))
    distil_bert_trainer = DistillationTrainer(model_init=student_init, teacher_model=teacher_model,
                                              args=student_training_args, train_dataset=clinc_enc["train"],
                                              eval_dataset=clinc_enc["validation"], compute_metrics=compute_metrics,
                                              tokenizer=student_tokenizer)

    distil_bert_trainer.train()
    distil_bert_trainer.push_to_hub("Training Completed")

    best_run = distil_bert_trainer.hyperparameter_search(n_trials=20, direction="maximize", hp_space=hp_space)
    print(best_run)
