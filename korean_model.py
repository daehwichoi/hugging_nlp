from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_path = "kyujinpy/Korean-OpenOrca-13B-v3"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    bert_model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

    text = "아 진짜 너무"
    input_ids = tokenizer(text, return_tensors='pt')['input_ids'].to(device)
    max_length = 128
    output = bert_model.generate(input_ids, max_length= max_length, do_sample= False)

    print(tokenizer.decode(output[0]))
