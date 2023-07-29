import torch
import logging
import transformers
from tqdm import tqdm
from fastapi import FastAPI
from time import perf_counter
from fastapi.logger import logger
from transformers import AutoTokenizer
from fastapi.websockets import WebSocket, WebSocketDisconnect


# model = "EleutherAI/gpt-j-6B"  # DONE
# model = "tiiuae/falcon-7b" # Takes a lot of time for inference
# model = "tiiuae/falcon-40b" # DONT TRY THIS
# model = meta-llama/Llama-2-7b
# model = meta-llama/Llama-2-13b
# model = meta-llama/Llama-2-70b # DONT TRY THIS
model = ""


class Inference:
    def __init__(self, model_name: str = None):
        self.loaded_models = {}
        if model_name:
            self.load(model_name)

    def load(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        if model_name not in self.loaded_models.keys():
            self.loaded_models[model_name] = pipeline

    def pop(self, model_name):
        self.loaded_models.pop(model_name)


# tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=model,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)
logger.setLevel(logging.WARN)
logger.setLevel(logging.ERROR)
logger.setLevel(logging.WARNING)
logger.setLevel(logging.CRITICAL)

app = FastAPI()


async def infer(prompt, ws: WebSocket):
    logger.info("Started Inference!")
    print("Started Inference!")
    sequences = pipeline(
        prompt,
        max_length=32768,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=pipeline.tokenizer.eos_token_id,
    )
    logger.info("Ended Inference")
    print("Ended Inference")
    await ws.send_json([seq["generated_text"] for seq in sequences])
    return [seq["generated_text"] for seq in sequences]


@app.get("")
async def index():
    return dict(status="working")


@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    while True:
        try:
            prompts = await ws.receive_json()
            for prompt in tqdm(prompts):
                s = perf_counter()
                await infer(prompt, ws)
                e = perf_counter()
                logger.info(f"Time Taken {e-s:.2f} seconds")
                print(f"Time Taken {e-s:.2f} seconds")
        except RuntimeError as e:
            logger.info(e)
            print(e)
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected!")
            print("WebSocket disconnected!")
            break
