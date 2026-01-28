from typing import cast

import torch
from torch._prims_common import DeviceLikeType
from transformers import AutoModelForCausalLM, AutoTokenizer

device = cast(DeviceLikeType, torch.device("mps"))

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype=torch.float16,
).to(device)


def main():
    print("Hello from knowledge-distillation-example!")


if __name__ == "__main__":
    main()
