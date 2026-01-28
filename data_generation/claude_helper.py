import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from config import (
    DATA_DIR,
    MERMAID_MAX_LABEL_WORDS,
    MERMAID_MAX_NODES,
    MERMAID_MIN_NODES,
    RAW_DATA_DIR,
    TARGET_SAMPLES,
    TRAIN_VAL_SPLIT,
    TRAINING_DATA_PATH,
    VALIDATION_DATA_PATH,
)

from .domains import COMPLEXITY_LEVELS, DOMAINS
from .validator import validate_mermaid

GENERATED_DATA_PATH = RAW_DATA_DIR / "claude_generated.jsonl"
PROGRESS_PATH = DATA_DIR / "progress.json"
SESSION_PATH = DATA_DIR / "session.json"
BATCH_FILE_PATH = DATA_DIR / "current_batch.json"


def output_json(data: Any) -> None:
    print(json.dumps(data, indent=2))


def load_session() -> dict:
    if SESSION_PATH.exists():
        with open(SESSION_PATH, "r") as f:
            return json.load(f)
    return {}


def save_session(session: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SESSION_PATH, "w") as f:
        json.dump(session, f, indent=2)


def load_progress() -> dict:
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, "r") as f:
            return json.load(f)
    return {
        "count": 0,
        "used_combinations": [],
        "domain_counts": {},
    }


def save_progress(progress: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)


def load_generated_samples() -> list[dict]:
    samples = []
    if GENERATED_DATA_PATH.exists():
        with open(GENERATED_DATA_PATH, "r") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
    return samples


def append_samples(samples: list[dict]) -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(GENERATED_DATA_PATH, "a") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


def cmd_rules(args: argparse.Namespace) -> None:
    rules = {
        "mermaid_constraints": {
            "min_nodes": MERMAID_MIN_NODES,
            "max_nodes": MERMAID_MAX_NODES,
            "max_label_words": MERMAID_MAX_LABEL_WORDS,
            "direction": "flowchart TD (top-down)",
            "node_ids": "Use A, B, C, etc.",
            "arrows": "Use --> for connections",
            "decisions": "Use {curly braces} for decision nodes",
            "edge_labels": "Use -->|Yes| or -->|No| for decision branches",
        },
        "output_format": {
            "description": "1-2 sentence natural language description",
            "mermaid": "Complete flowchart starting with 'flowchart TD'",
            "domain": "The domain category provided",
            "topic": "The specific topic provided",
            "complexity": "simple, moderate, or complex",
        },
        "complexity_levels": {
            "simple": "3-4 nodes",
            "moderate": "5-6 nodes",
            "complex": "7-8 nodes",
        },
        "example": {
            "description": "A customer submits a request which is reviewed and either approved for processing or rejected.",
            "mermaid": "flowchart TD\n    A[Submit Request] --> B[Review]\n    B --> C{Approved?}\n    C -->|Yes| D[Process]\n    C -->|No| E[Reject]",
            "domain": "business_process",
            "topic": "customer onboarding",
            "complexity": "moderate",
        },
        "tips": [
            "Include at least one decision point for moderate/complex diagrams",
            "Keep labels concise (1-4 words)",
            "Make descriptions natural and varied",
            "Ensure the flowchart accurately represents the description",
        ],
    }
    output_json(rules)


def cmd_status(args: argparse.Namespace) -> None:
    progress = load_progress()
    samples = load_generated_samples()

    actual_count = len(samples)

    if actual_count != progress.get("count", 0):
        progress["count"] = actual_count
        save_progress(progress)

    domain_counts = {}
    for sample in samples:
        domain = sample.get("domain", "unknown")
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    status = {
        "total_generated": actual_count,
        "target": TARGET_SAMPLES,
        "remaining": max(0, TARGET_SAMPLES - actual_count),
        "progress_percent": round(100 * actual_count / TARGET_SAMPLES, 1),
        "domain_distribution": domain_counts,
        "total_domains": len(DOMAINS),
        "total_topics": sum(len(topics) for topics in DOMAINS.values()),
        "data_file": str(GENERATED_DATA_PATH),
    }
    output_json(status)


def cmd_get_batch(args: argparse.Namespace) -> None:
    count = args.count
    progress = load_progress()
    used_combinations = set(tuple(c) for c in progress.get("used_combinations", []))

    tasks = []
    all_combinations = []

    for domain, topics in DOMAINS.items():
        for topic in topics:
            for complexity_name, complexity_config in COMPLEXITY_LEVELS.items():
                combo = (domain, topic, complexity_name)
                all_combinations.append(combo)

    available = [c for c in all_combinations if c not in used_combinations]

    if len(available) < count:
        available = all_combinations

    selected = random.sample(available, min(count, len(available)))

    for domain, topic, complexity_name in selected:
        complexity_config = COMPLEXITY_LEVELS[complexity_name]
        tasks.append({
            "domain": domain,
            "topic": topic,
            "complexity": complexity_name,
            "min_nodes": complexity_config["min_nodes"],
            "max_nodes": complexity_config["max_nodes"],
        })

    output_json({"tasks": tasks, "count": len(tasks)})


def cmd_validate(args: argparse.Namespace) -> None:
    mermaid_code = args.mermaid

    is_valid, message = validate_mermaid(mermaid_code, check_syntax=True)

    output_json({
        "valid": is_valid,
        "message": message,
    })


def cmd_save_batch(args: argparse.Namespace) -> None:
    file_path = Path(args.file)

    if not file_path.exists():
        output_json({
            "success": False,
            "error": f"File not found: {file_path}",
            "saved": 0,
            "failed": 0,
        })
        return

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        output_json({
            "success": False,
            "error": f"Invalid JSON: {e}",
            "saved": 0,
            "failed": 0,
        })
        return

    if isinstance(data, list):
        examples = data
    elif isinstance(data, dict) and "examples" in data:
        examples = data["examples"]
    else:
        output_json({
            "success": False,
            "error": "Expected JSON array or object with 'examples' key",
            "saved": 0,
            "failed": 0,
        })
        return

    saved = []
    failed = []

    for i, example in enumerate(examples):
        required_fields = ["description", "mermaid", "domain", "topic", "complexity"]
        missing = [f for f in required_fields if f not in example]
        if missing:
            failed.append({
                "index": i,
                "error": f"Missing fields: {missing}",
            })
            continue

        is_valid, message = validate_mermaid(example["mermaid"], check_syntax=True)
        if not is_valid:
            failed.append({
                "index": i,
                "error": f"Invalid Mermaid: {message}",
                "description": example.get("description", "")[:50] + "...",
            })
            continue

        saved.append(example)

    if saved:
        append_samples(saved)

        progress = load_progress()
        progress["count"] = progress.get("count", 0) + len(saved)

        used = progress.get("used_combinations", [])
        for ex in saved:
            combo = [ex["domain"], ex["topic"], ex["complexity"]]
            if combo not in used:
                used.append(combo)
        progress["used_combinations"] = used

        domain_counts = progress.get("domain_counts", {})
        for ex in saved:
            domain = ex["domain"]
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        progress["domain_counts"] = domain_counts

        save_progress(progress)

    output_json({
        "success": True,
        "saved": len(saved),
        "failed": len(failed),
        "failures": failed if failed else None,
        "total_generated": load_progress()["count"],
    })


def cmd_finalize(args: argparse.Namespace) -> None:
    samples = load_generated_samples()

    if not samples:
        output_json({
            "success": False,
            "error": "No samples found to finalize",
        })
        return

    random.shuffle(samples)
    split_idx = int(len(samples) * TRAIN_VAL_SPLIT)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    TRAINING_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(TRAINING_DATA_PATH, "w") as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + "\n")

    with open(VALIDATION_DATA_PATH, "w") as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + "\n")

    output_json({
        "success": True,
        "total_samples": len(samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "train_file": str(TRAINING_DATA_PATH),
        "val_file": str(VALIDATION_DATA_PATH),
    })


def cmd_session(args: argparse.Namespace) -> None:
    action = args.action

    if action == "start":
        session = {
            "target": args.target,
            "batch_size": args.batch_size,
            "start_count": len(load_generated_samples()),
            "iterations": 0,
        }
        save_session(session)
        output_json({
            "action": "started",
            "session_target": args.target,
            "batch_size": args.batch_size,
            "current_total": session["start_count"],
            "message": f"Session started. Run 'session next' to begin generating.",
        })

    elif action == "next":
        session = load_session()
        if not session:
            output_json({
                "error": "No active session. Run 'session start --target N' first.",
            })
            return

        current_total = len(load_generated_samples())
        session_generated = current_total - session.get("start_count", 0)
        session_target = session.get("target", 500)
        session_remaining = max(0, session_target - session_generated)
        batch_size = session.get("batch_size", 10)

        if session_remaining == 0:
            output_json({
                "action": "complete",
                "session_target": session_target,
                "session_generated": session_generated,
                "total_generated": current_total,
                "message": "Session target reached! Run 'finalize' to create train/val split.",
            })
            return

        actual_batch_size = min(batch_size, session_remaining)
        progress = load_progress()
        used_combinations = set(tuple(c) for c in progress.get("used_combinations", []))

        all_combinations = []
        for domain, topics in DOMAINS.items():
            for topic in topics:
                for complexity_name, complexity_config in COMPLEXITY_LEVELS.items():
                    combo = (domain, topic, complexity_name)
                    all_combinations.append(combo)

        available = [c for c in all_combinations if c not in used_combinations]
        if len(available) < actual_batch_size:
            available = all_combinations

        selected = random.sample(available, min(actual_batch_size, len(available)))

        tasks = []
        for domain, topic, complexity_name in selected:
            complexity_config = COMPLEXITY_LEVELS[complexity_name]
            tasks.append({
                "domain": domain,
                "topic": topic,
                "complexity": complexity_name,
                "min_nodes": complexity_config["min_nodes"],
                "max_nodes": complexity_config["max_nodes"],
            })

        session["iterations"] = session.get("iterations", 0) + 1
        save_session(session)

        output_json({
            "action": "next",
            "iteration": session["iterations"],
            "progress": {
                "session_generated": session_generated,
                "session_target": session_target,
                "session_remaining": session_remaining,
                "session_percent": round(100 * session_generated / session_target, 1),
                "total_generated": current_total,
                "overall_target": TARGET_SAMPLES,
            },
            "batch": {
                "count": len(tasks),
                "tasks": tasks,
            },
            "save_to": str(BATCH_FILE_PATH),
            "instructions": [
                f"Generate {len(tasks)} examples for the tasks above.",
                f"Write the JSON array to: {BATCH_FILE_PATH}",
                f"Then run: python -m data_generation.claude_helper save-batch --file {BATCH_FILE_PATH}",
                "Then run: python -m data_generation.claude_helper session next",
            ],
        })

    elif action == "status":
        session = load_session()
        if not session:
            output_json({
                "error": "No active session. Run 'session start --target N' first.",
            })
            return

        current_total = len(load_generated_samples())
        session_generated = current_total - session.get("start_count", 0)
        session_target = session.get("target", 500)

        output_json({
            "session_target": session_target,
            "session_generated": session_generated,
            "session_remaining": max(0, session_target - session_generated),
            "session_percent": round(100 * session_generated / session_target, 1),
            "iterations": session.get("iterations", 0),
            "batch_size": session.get("batch_size", 10),
            "total_generated": current_total,
            "overall_target": TARGET_SAMPLES,
        })

    elif action == "end":
        if SESSION_PATH.exists():
            SESSION_PATH.unlink()
        output_json({
            "action": "ended",
            "message": "Session ended. Run 'status' to see overall progress.",
        })


def main():
    parser = argparse.ArgumentParser(
        description="CLI helper for Claude Code to generate Mermaid training data"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("rules", help="Print generation rules")

    subparsers.add_parser("status", help="Show generation progress")

    batch_parser = subparsers.add_parser("get-batch", help="Get batch of tasks")
    batch_parser.add_argument(
        "--count", type=int, default=10, help="Number of tasks to get"
    )

    validate_parser = subparsers.add_parser("validate", help="Validate Mermaid code")
    validate_parser.add_argument(
        "--mermaid", type=str, required=True, help="Mermaid code to validate"
    )

    save_parser = subparsers.add_parser("save-batch", help="Save batch of examples")
    save_parser.add_argument(
        "--file", type=str, required=True, help="Path to JSON file with examples"
    )

    subparsers.add_parser("finalize", help="Create train/val split")

    session_parser = subparsers.add_parser(
        "session", help="Manage generation session for looped generation"
    )
    session_parser.add_argument(
        "action",
        choices=["start", "next", "status", "end"],
        help="Session action: start, next, status, or end",
    )
    session_parser.add_argument(
        "--target", type=int, default=500, help="Target samples for this session"
    )
    session_parser.add_argument(
        "--batch-size", type=int, default=10, help="Samples per batch"
    )

    args = parser.parse_args()

    commands = {
        "rules": cmd_rules,
        "status": cmd_status,
        "get-batch": cmd_get_batch,
        "validate": cmd_validate,
        "save-batch": cmd_save_batch,
        "finalize": cmd_finalize,
        "session": cmd_session,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
