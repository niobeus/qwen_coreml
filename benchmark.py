import numpy as np

import time

import torch
from transformers.trainer_utils import set_seed

from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers import (
    AutoTokenizer,
)
from coremltools.models import MLModel


MODEL_ID: str = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
attn_impl = "eager"
model = Qwen2ForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation=attn_impl,
).eval()

# # Prepare inputs
DUMMY_INPUT = "我"
n_tokens = 1024
context_str = DUMMY_INPUT * n_tokens
inputs = tokenizer([context_str], return_tensors="pt")
inputs = inputs.to(model.device)

# warmup
_ = model.forward(**inputs)

n_cycles = 5
start_time = time.time()
for _ in range(n_cycles):
    pred = model.forward(**inputs)
time_cost = time.time() - start_time

print("Torch model:", time_cost / n_cycles / n_tokens)

model = MLModel("StatefulQwen2.51.5BInstructFp16.mlpackage")
description = model.get_spec().description
tokenizer_path: str = description.metadata.userDefined["co.huggingface.exporters.name"]
tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

kv_cache_state = model.make_state()
prompt_tokens: np.ndarray = tokenizer(
    context_str, return_tensors="np"
).input_ids.astype(np.int32)
num_past_tokens: int = prompt_tokens.shape[-1]
causal_mask: np.ndarray = np.triu(
    np.full(
        (1, 1, prompt_tokens.shape[-1], 0 + prompt_tokens.shape[-1]),
        fill_value=-np.inf if num_past_tokens == 0 else 0,
    ),
    k=1,
).astype(np.float16)
_ = model.predict(
    data={"inputIds": prompt_tokens, "causalMask": causal_mask},
    state=kv_cache_state,
)

start_time = time.time()
for _ in range(n_cycles):
    pred = model.predict(
        data={"inputIds": prompt_tokens, "causalMask": causal_mask},
        state=kv_cache_state,
    )
time_cost = time.time() - start_time

print("CoreML model:", time_cost / n_cycles / n_tokens)
