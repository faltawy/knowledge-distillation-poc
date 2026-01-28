import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import LORA_OUTPUT_DIR, STUDENT_MODEL_ID


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(merge_adapter: bool = False):
    device = get_device()
    print(f"Loading model on {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(LORA_OUTPUT_DIR),
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_ID,
        trust_remote_code=True,
        dtype=torch.float16,
        device_map={"": device},
    )

    model = PeftModel.from_pretrained(base_model, str(LORA_OUTPUT_DIR))

    if merge_adapter:
        print("Merging adapter weights...")
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def generate_mermaid(
    description: str,
    model=None,
    tokenizer=None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    if model is None or tokenizer is None:
        model, tokenizer = load_model()

    messages = [
        {
            "role": "user",
            "content": f"Convert this process description to a Mermaid flowchart:\n\n{description}",
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    response = full_response[
        len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)) :
    ]
    return response.strip()


def run_examples():
    print("Loading fine-tuned model...")
    model, tokenizer = load_model()

    examples = [
        "A user submits a support ticket, which is reviewed by an agent. If the issue is simple, it's resolved immediately. Otherwise, it's escalated to a specialist who investigates and resolves it.",
        "An employee requests time off. The manager reviews the request and either approves or denies it. If approved, HR updates the records.",
        "A customer places an order online. The system checks inventory. If in stock, the order is fulfilled and shipped. If out of stock, the customer is notified.",
    ]

    print("\n" + "=" * 60)
    print("INFERENCE EXAMPLES")
    print("=" * 60)

    for i, desc in enumerate(examples, 1):
        print(f"\n--- Example {i} ---")
        print(f"Description: {desc[:100]}...")
        print("\nGenerated Mermaid:")
        result = generate_mermaid(desc, model, tokenizer)
        print(result)
        print()


def main():
    run_examples()


if __name__ == "__main__":
    main()
