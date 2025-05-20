import numpy as np
import time
from tqdm import tqdm
import torch
from transformers.trainer_utils import set_seed
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers import AutoTokenizer
from coremltools.models import MLModel


def load_torch_model(model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2ForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    ).eval()
    return model, tokenizer


def load_coreml_model(model_path: str = "StatefulQwen2.51.5BInstructFp16.mlpackage"):

    model = MLModel(model_path)
    description = model.get_spec().description
    tokenizer_path: str = description.metadata.userDefined[
        "co.huggingface.exporters.name"
    ]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer


# Load models
torch_model, torch_tokenizer = load_torch_model()
coreml_model, coreml_tokenizer = load_coreml_model()

# Prepare inputs
context_str = "Hello!"
inputs = torch_tokenizer([context_str], return_tensors="pt")
inputs = inputs.to(torch_model.device)
past_key_values = None

# warmup
with torch.inference_mode():
    outputs = torch_model.forward(**inputs)
    past_key_values = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1)
    inputs = {"input_ids": next_token.unsqueeze(0)}

n_cycles = 1024
torch_outputs = []
with torch.inference_mode():
    start_time = time.time()
    for _ in tqdm(range(n_cycles)):
        outputs = torch_model.forward(**inputs, past_key_values=past_key_values)
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1)
        inputs = {"input_ids": next_token.unsqueeze(0)}
        torch_outputs.append(next_token.item())
time_cost = time.time() - start_time

print("Torch model:", time_cost)

kv_cache_state = coreml_model.make_state()
prompt_tokens: np.ndarray = coreml_tokenizer(
    context_str, return_tensors="np"
).input_ids.astype(np.int32)
num_past_tokens: int = prompt_tokens.shape[-1]
causal_mask: np.ndarray = np.triu(
    np.full(
        (1, 1, prompt_tokens.shape[-1], prompt_tokens.shape[-1]),
        fill_value=-np.inf if num_past_tokens == 0 else 0,
    ),
    k=1,
).astype(np.float16)

outputs = coreml_model.predict(
    data={"inputIds": prompt_tokens, "causalMask": causal_mask},
    state=kv_cache_state,
)
next_token = np.argmax(outputs["logits"][0][-1], axis=-1)
prompt_tokens = next_token.reshape(1, 1).astype(np.int32)

kv_cache_state = coreml_model.make_state()
coreml_outputs = []
start_time = time.time()
for num_past_tokens in tqdm(range(n_cycles)):
    causal_mask: np.ndarray = np.triu(
        np.full(
            (1, 1, prompt_tokens.shape[-1], num_past_tokens + prompt_tokens.shape[-1]),
            fill_value=-np.inf if num_past_tokens == 0 else 0,
        ),
        k=1,
    ).astype(np.float16)
    outputs = coreml_model.predict(
        data={"inputIds": prompt_tokens, "causalMask": causal_mask},
        state=kv_cache_state,
    )
    next_token = np.argmax(outputs["logits"][0][-1], axis=-1)
    prompt_tokens = next_token.reshape(1, 1).astype(np.int32)
    coreml_outputs.append(next_token.item())

time_cost = time.time() - start_time

print("CoreML model:", time_cost)

print("Torch output: ", torch_tokenizer.decode(torch_outputs))
print("CoreML output: ", coreml_tokenizer.decode(coreml_outputs))
