import pandas as pd
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import torch
import torch.nn.functional as F


# 조건부 확률까지 가면 너무 확률값이 작다.
def log_probs_from_logits(logits, labels):
    logp = F.log_softmax(logits, dim=-1)
    logp_label = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logp_label


def sequence_logprob(model, labels, input_len=0):
    with torch.no_grad():
        output = model(labels)
        log_probs = log_probs_from_logits(output.logits[:, :-1, :], output.logits[:, 1:])
        seq_log_prob = torch.sum(log_probs[:, input_len:])
    return seq_log_prob.cpu().numpy()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    text = "I want to go"

    model_name = 'gpt2-xl'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    n_steps = 10
    choices_per_step = 5
    max_length = 128
    output = model.generate(input_ids, max_length=max_length, do_sample=True, top_k=50)
    # logp = sequence_logprob(model, output, input_len=(input_ids[0]))
    # print("log probability : {logp:.2f]")
    print(tokenizer.decode(output[0]))

    # iterations = []
    #
    #
    # with torch.no_grad():
    #     for _ in range(n_steps):
    #         iteration = dict()
    #         iteration["Input"] = tokenizer.decode(input_ids[0])
    #         output = model(input_ids=input_ids)
    #
    #         next_token_logits = output.logits[0, -1, :]
    #         next_token_probs = torch.softmax(next_token_logits, dim=-1)
    #         sorted_ids = torch.argsort(next_token_probs, dim=-1, descending=True)
    #
    #         for choice_idx in range(choices_per_step):
    #             token_id = sorted_ids[choice_idx]
    #             token_prob = next_token_probs[token_id].cpu().numpy()
    #             token_choice = f"{tokenizer.decode(token_id)} ({100 * token_prob:.2f}%)"
    #             iteration[f"Choice {choice_idx + 1}"] = token_choice
    #         input_ids = torch.cat([input_ids, sorted_ids[None, 0, None]], dim=-1)
    #         iterations.append(iteration)
    # data = pd.DataFrame(iterations)
    # print(data)
