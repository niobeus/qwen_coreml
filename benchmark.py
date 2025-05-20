import numpy as np
import time
from tqdm import tqdm
import torch
import argparse
from typing import Dict
from transformers.trainer_utils import set_seed
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers import AutoTokenizer
from coremltools.models import MLModel
from coremltools.models.model import MLState


def load_torch_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2ForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    ).eval()
    return model, tokenizer


def load_coreml_model(model_path: str):

    model = MLModel(model_path)
    description = model.get_spec().description
    tokenizer_path: str = description.metadata.userDefined[
        "co.huggingface.exporters.name"
    ]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer


def coreml_predict(
    model: MLModel,
    input_ids: np.ndarray,
    kv_cache_state: MLState,
    num_past_tokens: int,
):
    causal_mask: np.ndarray = np.triu(
        np.full(
            (1, 1, input_ids.shape[-1], num_past_tokens + input_ids.shape[-1]),
            fill_value=-np.inf if num_past_tokens == 0 else 0,
        ),
        k=1,
    ).astype(np.float16)

    outputs: Dict[str, np.ndarray] = model.predict(
        data={"inputIds": input_ids, "causalMask": causal_mask},
        state=kv_cache_state,
    )
    return outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch and CoreML models")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, my dear friend",
        help="Input prompt for text generation",
    )
    parser.add_argument(
        "--torch-model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Path or HuggingFace ID for the PyTorch model",
    )
    parser.add_argument(
        "--coreml-model",
        type=str,
        default="StatefulQwen2.51.5BInstructFp16.mlpackage",
        help="Path to the CoreML model package",
    )
    parser.add_argument(
        "--n-cycles", type=int, default=128, help="Number of generation cycles"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    torch_model, torch_tokenizer = load_torch_model(args.torch_model)
    coreml_model, coreml_tokenizer = load_coreml_model(args.coreml_model)

    inputs = torch_tokenizer([args.prompt], return_tensors="pt")
    inputs = inputs.to(torch_model.device)
    past_key_values = None

    # warmup and kv cache filling
    with torch.inference_mode():
        outputs = torch_model.forward(**inputs)
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1)
        inputs = {"input_ids": next_token.unsqueeze(0)}

    torch_outputs = []
    with torch.inference_mode():
        start_time = time.time()
        for _ in tqdm(range(args.n_cycles)):
            outputs = torch_model.forward(**inputs, past_key_values=past_key_values)
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1)
            inputs = {"input_ids": next_token.unsqueeze(0)}
            torch_outputs.append(next_token.item())
    torch_time_cost = time.time() - start_time

    kv_cache_state = coreml_model.make_state()
    prompt_tokens: np.ndarray = coreml_tokenizer(
        args.prompt, return_tensors="np"
    ).input_ids.astype(np.int32)
    num_past_tokens: int = prompt_tokens.shape[-1]

    outputs = coreml_predict(coreml_model, prompt_tokens, kv_cache_state, 0)
    next_token = np.argmax(outputs["logits"][0][-1], axis=-1)
    input_ids = np.array([[next_token]], dtype=np.int32)

    coreml_outputs = []
    start_time = time.time()
    for num_gen_tokens in tqdm(range(args.n_cycles)):
        outputs = coreml_predict(
            coreml_model, input_ids, kv_cache_state, num_past_tokens + num_gen_tokens
        )
        next_token = np.argmax(outputs["logits"][0][-1], axis=-1)
        input_ids = np.array([[next_token]], dtype=np.int32)
        coreml_outputs.append(next_token.item())

    coreml_time_cost = time.time() - start_time

    print("Time cost of Torch model:", torch_time_cost)
    print("Time cost of CoreML model:", coreml_time_cost)

    print("Torch output: ", torch_tokenizer.decode(torch_outputs))
    print("CoreML output: ", coreml_tokenizer.decode(coreml_outputs))
