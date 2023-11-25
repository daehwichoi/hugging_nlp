
import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    emotion = load_dataset("emotion")

    model_ckpt = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    encoded_data = emotion.map(lambda row: tokenizer(row["text"], padding=True, truncation=True), batched=True,
                               batch_size=None)

    num_labels = 6
    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels).to(device)

    batch_size = 64
    logging_steps = len(encoded_data["train"]) // batch_size
    model_name = f"{model_ckpt}-finetuned-emotion"
    training_args = TrainingArguments(output_dir=model_name, num_train_epochs=2, learning_rate=2e-5,
                                      per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
                                      weight_decay=0.01, evaluation_strategy="epoch", disable_tqdm=False,
                                      logging_steps=logging_steps, push_to_hub=True, save_strategy="epoch",
                                      load_best_model_at_end=True, log_level='error')

    trainer = Trainer(model=model, args=training_args, compute_metrics=compute_metrics,
                      train_dataset=encoded_data["train"], eval_dataset=encoded_data["validation"], tokenizer=tokenizer)
    trainer.train()