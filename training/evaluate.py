import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    LORA_OUTPUT_DIR,
    STUDENT_MODEL_ID,
    VALIDATION_DATA_PATH,
)
from data_generation.validator import (
    extract_nodes,
    validate_constraints,
    validate_mermaid_syntax,
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_validation_data(path: Optional[Path] = None, limit: Optional[int] = None) -> list[dict]:
    path = path or VALIDATION_DATA_PATH
    samples = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
                if limit and len(samples) >= limit:
                    break
    return samples


def load_base_model(device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(
        STUDENT_MODEL_ID,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_ID,
        trust_remote_code=True,
        dtype=torch.float16,
        device_map={"": device},
    )
    model.eval()
    return model, tokenizer


def load_finetuned_model(device: torch.device):
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
    model.eval()
    return model, tokenizer


def extract_mermaid_from_response(response: str) -> str:
    mermaid_block = re.search(r"```mermaid\s*\n(.*?)```", response, re.DOTALL)
    if mermaid_block:
        return mermaid_block.group(1).strip()

    generic_block = re.search(r"```\s*\n?(flowchart\s+(?:TD|TB|BT|LR|RL).*?)```", response, re.DOTALL)
    if generic_block:
        return generic_block.group(1).strip()

    return response.strip()


def generate_mermaid(
    description: str,
    model,
    tokenizer,
    max_new_tokens: int = 512,
) -> str:
    messages = [
        {
            "role": "user",
            "content": f"Convert this process description to a Mermaid flowchart. Output ONLY the Mermaid code starting with 'flowchart TD', no explanations.\n\n{description}",
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
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response[
        len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)) :
    ]
    return extract_mermaid_from_response(response)


def extract_edges(mermaid_code: str) -> list[tuple[str, str, Optional[str]]]:
    edges = []
    lines = mermaid_code.strip().split("\n")

    edge_pattern = r"([A-Za-z][A-Za-z0-9]*)\s*-->\s*(?:\|([^|]+)\|)?\s*([A-Za-z][A-Za-z0-9]*)"

    for line in lines[1:]:
        matches = re.findall(edge_pattern, line)
        for from_node, label, to_node in matches:
            edges.append((from_node, to_node, label if label else None))

    return edges


def calculate_bleu(generated: str, reference: str, max_n: int = 4) -> float:
    def tokenize(text: str) -> list[str]:
        return re.findall(r"[A-Za-z]+|\[|\]|\{|\}|\(|\)|-->|\||[0-9]+", text.lower())

    def get_ngrams(tokens: list[str], n: int) -> dict[tuple, int]:
        ngrams = defaultdict(int)
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams[ngram] += 1
        return ngrams

    gen_tokens = tokenize(generated)
    ref_tokens = tokenize(reference)

    if len(gen_tokens) == 0:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        gen_ngrams = get_ngrams(gen_tokens, n)
        ref_ngrams = get_ngrams(ref_tokens, n)

        if not gen_ngrams:
            precisions.append(0.0)
            continue

        clipped_count = sum(
            min(count, ref_ngrams.get(ngram, 0))
            for ngram, count in gen_ngrams.items()
        )
        total_count = sum(gen_ngrams.values())

        precision = clipped_count / total_count if total_count > 0 else 0.0
        precisions.append(precision)

    if any(p == 0 for p in precisions):
        return 0.0

    import math
    log_precision = sum(math.log(p) for p in precisions) / len(precisions)

    bp = 1.0 if len(gen_tokens) >= len(ref_tokens) else math.exp(1 - len(ref_tokens) / len(gen_tokens))

    return bp * math.exp(log_precision)


def calculate_edit_distance(s1: str, s2: str) -> float:
    if s1 == s2:
        return 0.0

    len1, len2 = len(s1), len(s2)
    if len1 == 0:
        return 1.0
    if len2 == 0:
        return 1.0

    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[len1][len2] / max(len1, len2)


def evaluate_single(
    description: str,
    generated: str,
    ground_truth: str,
    check_syntax: bool = True,
) -> dict:
    syntax_valid = False
    if check_syntax:
        syntax_valid, _ = validate_mermaid_syntax(generated)

    constraints_met, _ = validate_constraints(generated)

    exact_match = generated.strip() == ground_truth.strip()

    bleu = calculate_bleu(generated, ground_truth)

    edit_distance = calculate_edit_distance(generated.strip(), ground_truth.strip())

    gen_nodes = extract_nodes(generated)
    ref_nodes = extract_nodes(ground_truth)
    gen_edges = extract_edges(generated)
    ref_edges = extract_edges(ground_truth)

    has_decision = any(node["type"] == "diamond" for node in gen_nodes)
    has_edge_labels = any(edge[2] is not None for edge in gen_edges)

    return {
        "syntax_valid": syntax_valid,
        "constraints_met": constraints_met,
        "exact_match": exact_match,
        "bleu": bleu,
        "edit_distance": edit_distance,
        "node_count": len(gen_nodes),
        "ref_node_count": len(ref_nodes),
        "edge_count": len(gen_edges),
        "ref_edge_count": len(ref_edges),
        "has_decision": has_decision,
        "has_edge_labels": has_edge_labels,
    }


def evaluate_model(
    model,
    tokenizer,
    samples: list[dict],
    model_name: str = "Model",
    check_syntax: bool = True,
    verbose: bool = False,
) -> dict:
    results = []
    complexity_results = defaultdict(list)

    print(f"\nEvaluating {model_name} on {len(samples)} samples...")

    for i, sample in enumerate(samples):
        if verbose or (i + 1) % 10 == 0:
            print(f"  Processing {i + 1}/{len(samples)}...")

        generated = generate_mermaid(sample["description"], model, tokenizer)
        metrics = evaluate_single(
            sample["description"],
            generated,
            sample["mermaid"],
            check_syntax=check_syntax,
        )
        metrics["complexity"] = sample.get("complexity", "unknown")
        metrics["domain"] = sample.get("domain", "unknown")

        results.append(metrics)
        complexity_results[metrics["complexity"]].append(metrics)

    n = len(results)
    aggregate = {
        "syntax_valid": sum(r["syntax_valid"] for r in results) / n * 100,
        "constraints_met": sum(r["constraints_met"] for r in results) / n * 100,
        "exact_match": sum(r["exact_match"] for r in results) / n * 100,
        "bleu": sum(r["bleu"] for r in results) / n,
        "edit_distance": sum(r["edit_distance"] for r in results) / n,
        "avg_node_count": sum(r["node_count"] for r in results) / n,
        "avg_edge_count": sum(r["edge_count"] for r in results) / n,
    }

    by_complexity = {}
    for complexity, comp_results in complexity_results.items():
        cn = len(comp_results)
        by_complexity[complexity] = {
            "count": cn,
            "syntax_valid": sum(r["syntax_valid"] for r in comp_results) / cn * 100,
            "constraints_met": sum(r["constraints_met"] for r in comp_results) / cn * 100,
        }

    return {
        "aggregate": aggregate,
        "by_complexity": by_complexity,
        "raw_results": results,
    }


def print_comparison_table(finetuned_results: dict, base_results: dict) -> None:
    ft = finetuned_results["aggregate"]
    base = base_results["aggregate"]

    print("\n" + "=" * 80)
    print("                        MODEL EVALUATION RESULTS")
    print("=" * 80)
    print()
    print(f"{'Metric':<25} {'Fine-tuned':>12} {'Base Model':>12} {'Improvement':>12}")
    print("-" * 80)

    for metric, label in [
        ("syntax_valid", "Syntax Valid"),
        ("constraints_met", "Constraints Met"),
        ("exact_match", "Exact Match"),
    ]:
        ft_val = ft[metric]
        base_val = base[metric]
        diff = ft_val - base_val
        print(f"{label:<25} {ft_val:>11.1f}% {base_val:>11.1f}% {diff:>+11.1f}%")

    print(f"{'BLEU Score':<25} {ft['bleu']:>12.2f} {base['bleu']:>12.2f} {ft['bleu'] - base['bleu']:>+12.2f}")

    print(f"{'Avg Edit Distance':<25} {ft['edit_distance']:>12.2f} {base['edit_distance']:>12.2f} {ft['edit_distance'] - base['edit_distance']:>+12.2f}")

    print()
    print("By Complexity (Constraints Met):")
    print("-" * 80)

    for complexity in ["simple", "moderate", "complex"]:
        ft_comp = finetuned_results["by_complexity"].get(complexity, {})
        base_comp = base_results["by_complexity"].get(complexity, {})

        if ft_comp and base_comp:
            ft_val = ft_comp["constraints_met"]
            base_val = base_comp["constraints_met"]
            diff = ft_val - base_val
            print(f"  {complexity.capitalize():<22} {ft_val:>11.1f}% {base_val:>11.1f}% {diff:>+11.1f}%")

    print("=" * 80)


def print_single_model_results(results: dict, model_name: str) -> None:
    agg = results["aggregate"]

    print("\n" + "=" * 60)
    print(f"           {model_name.upper()} EVALUATION RESULTS")
    print("=" * 60)
    print()
    print(f"{'Metric':<30} {'Value':>15}")
    print("-" * 60)
    print(f"{'Syntax Valid':<30} {agg['syntax_valid']:>14.1f}%")
    print(f"{'Constraints Met':<30} {agg['constraints_met']:>14.1f}%")
    print(f"{'Exact Match':<30} {agg['exact_match']:>14.1f}%")
    print(f"{'BLEU Score':<30} {agg['bleu']:>15.2f}")
    print(f"{'Avg Edit Distance':<30} {agg['edit_distance']:>15.2f}")
    print(f"{'Avg Node Count':<30} {agg['avg_node_count']:>15.1f}")
    print(f"{'Avg Edge Count':<30} {agg['avg_edge_count']:>15.1f}")

    print()
    print("By Complexity:")
    print("-" * 60)

    for complexity in ["simple", "moderate", "complex"]:
        comp_data = results["by_complexity"].get(complexity, {})
        if comp_data:
            print(f"  {complexity.capitalize():<27} {comp_data['constraints_met']:>14.1f}% ({comp_data['count']} samples)")

    print("=" * 60)


def compare_models(
    samples: list[dict],
    check_syntax: bool = True,
    verbose: bool = False,
) -> tuple[dict, dict]:
    device = get_device()
    print(f"Using device: {device}")

    print("\nLoading fine-tuned model...")
    ft_model, ft_tokenizer = load_finetuned_model(device)
    finetuned_results = evaluate_model(
        ft_model, ft_tokenizer, samples,
        model_name="Fine-tuned",
        check_syntax=check_syntax,
        verbose=verbose,
    )

    del ft_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\nLoading base model...")
    base_model, base_tokenizer = load_base_model(device)
    base_results = evaluate_model(
        base_model, base_tokenizer, samples,
        model_name="Base",
        check_syntax=check_syntax,
        verbose=verbose,
    )

    print_comparison_table(finetuned_results, base_results)

    return finetuned_results, base_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Mermaid generation models")
    parser.add_argument(
        "--model",
        choices=["both", "finetuned", "base"],
        default="both",
        help="Which model(s) to evaluate (default: both)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--no-syntax-check",
        action="store_true",
        help="Skip mermaid.ink syntax validation (faster)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress for each sample",
    )

    args = parser.parse_args()

    samples = load_validation_data(limit=args.samples)
    print(f"Loaded {len(samples)} validation samples")

    check_syntax = not args.no_syntax_check
    device = get_device()

    if args.model == "both":
        compare_models(samples, check_syntax=check_syntax, verbose=args.verbose)

    elif args.model == "finetuned":
        print(f"\nUsing device: {device}")
        print("\nLoading fine-tuned model...")
        model, tokenizer = load_finetuned_model(device)
        results = evaluate_model(
            model, tokenizer, samples,
            model_name="Fine-tuned",
            check_syntax=check_syntax,
            verbose=args.verbose,
        )
        print_single_model_results(results, "Fine-tuned Model")

    elif args.model == "base":
        print(f"\nUsing device: {device}")
        print("\nLoading base model...")
        model, tokenizer = load_base_model(device)
        results = evaluate_model(
            model, tokenizer, samples,
            model_name="Base",
            check_syntax=check_syntax,
            verbose=args.verbose,
        )
        print_single_model_results(results, "Base Model")


if __name__ == "__main__":
    main()
