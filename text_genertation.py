import pandas as pd
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import torch

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    text = "This is my last"

    model_name = 'gpt2-xl'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    iterations = []

    n_steps = 10
    choices_per_step = 5

    with torch.no_grad():
        for _ in range(n_steps):
            iteration = dict()
            iteration["Input"] = tokenizer.decode(input_ids[0])
            output = model(input_ids=input_ids)

            next_token_logits = output.logits[0, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            sorted_ids = torch.argsort(next_token_probs, dim=-1, descending=True)

            for choice_idx in range(choices_per_step):
                token_id = sorted_ids[choice_idx]
                token_prob = next_token_probs[token_id].cpu().numpy()
                token_choice = f"{tokenizer.decode(token_id)} ({100 * token_prob:.2f}%)"
                iteration[f"Choice {choice_idx + 1}"] = token_choice
            input_ids = torch.cat([input_ids, sorted_ids[None, 0, None]], dim=-1)
            iterations.append(iteration)
    data = pd.DataFrame(iterations)
    print(data)
