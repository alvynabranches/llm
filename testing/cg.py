import torch
from time import perf_counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompts import prompts
from sb import send_message

MODEL_NAME = "Salesforce/codegen2-16B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, trust_remote_code=True, revision="main", device_map="auto")


def generate(prompt: str):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    generated_ids = model.generate(input_ids, max_length=4096)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

try:
    for prompt in prompts:
        s = perf_counter()
        output = generate(prompt)
        e = perf_counter()
        send_message(prompt, output, MODEL_NAME, e-s)
except Exception as e:
    send_message(e)
