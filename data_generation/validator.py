import re
from typing import Tuple

import mermaid as md
from mermaid.graph import Graph

from config import MERMAID_MAX_NODES, MERMAID_MIN_NODES, MERMAID_MAX_LABEL_WORDS


def validate_mermaid_syntax(mermaid_code: str) -> Tuple[bool, str]:
    try:
        graph = Graph("validation_test", mermaid_code)
        diagram = md.Mermaid(graph)
        response = diagram.svg_response
        if response.status_code == 200:
            return True, "Valid syntax"
        else:
            return False, f"Mermaid syntax error (HTTP {response.status_code})"
    except Exception as e:
        return False, f"Mermaid validation error: {str(e)}"


def validate_constraints(mermaid_code: str) -> Tuple[bool, str]:
    if not mermaid_code or not mermaid_code.strip():
        return False, "Empty mermaid code"

    lines = mermaid_code.strip().split("\n")

    first_line = lines[0].strip().lower()
    if not first_line.startswith("flowchart td"):
        return False, "Must start with 'flowchart TD'"

    nodes = set()
    node_pattern = r"([A-Za-z][A-Za-z0-9]*)\s*[\[\{\(]([^\]\}\)]+)[\]\}\)]"

    for line in lines[1:]:
        matches = re.findall(node_pattern, line)
        for node_id, label in matches:
            nodes.add(node_id.upper())

            words = label.strip().split()
            if len(words) > MERMAID_MAX_LABEL_WORDS:
                return False, f"Label '{label}' has {len(words)} words (max {MERMAID_MAX_LABEL_WORDS})"

    if len(nodes) < MERMAID_MIN_NODES:
        return False, f"Too few nodes: {len(nodes)} (min {MERMAID_MIN_NODES})"
    if len(nodes) > MERMAID_MAX_NODES:
        return False, f"Too many nodes: {len(nodes)} (max {MERMAID_MAX_NODES})"

    arrow_pattern = r"-->"
    has_arrows = any(re.search(arrow_pattern, line) for line in lines[1:])
    if not has_arrows:
        return False, "No valid arrows (-->) found"

    return True, "Valid"


def validate_mermaid(mermaid_code: str, check_syntax: bool = True) -> Tuple[bool, str]:
    is_valid, error = validate_constraints(mermaid_code)
    if not is_valid:
        return False, error

    if check_syntax:
        is_valid, error = validate_mermaid_syntax(mermaid_code)
        if not is_valid:
            return False, error

    return True, "Valid"


def extract_nodes(mermaid_code: str) -> list[dict]:
    nodes = []
    lines = mermaid_code.strip().split("\n")

    patterns = [
        (r"([A-Za-z][A-Za-z0-9]*)\s*\[([^\]]+)\]", "rectangle"),
        (r"([A-Za-z][A-Za-z0-9]*)\s*\{([^\}]+)\}", "diamond"),
        (r"([A-Za-z][A-Za-z0-9]*)\s*\(([^\)]+)\)", "rounded"),
    ]

    seen = set()
    for line in lines[1:]:
        for pattern, node_type in patterns:
            matches = re.findall(pattern, line)
            for node_id, label in matches:
                if node_id not in seen:
                    nodes.append({
                        "id": node_id,
                        "label": label.strip(),
                        "type": node_type,
                    })
                    seen.add(node_id)

    return nodes
