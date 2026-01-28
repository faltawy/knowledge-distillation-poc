import sys


def print_welcome():
    print("\n" + "=" * 60)
    print("  Mermaid Flowchart Generator")
    print("=" * 60)
    print("\nDescribe a process and I'll generate a flowchart for it.")
    print("\nCommands:")
    print("  /help  - Show this help message")
    print("  /quit  - Exit the chat")
    print("\nTip: Press Enter twice to submit your description.")
    print("=" * 60 + "\n")


def print_help():
    print("\n--- Help ---")
    print("Describe a process or workflow in natural language.")
    print("Example: 'User logs in, system verifies credentials, grants access'")
    print("\nCommands:")
    print("  /help  - Show this help")
    print("  /quit  - Exit")
    print()


def read_multiline_input() -> str:
    lines = []
    while True:
        try:
            line = input()
            if line == "" and lines:
                break
            lines.append(line)
        except EOFError:
            break
    return "\n".join(lines).strip()


def format_mermaid(code: str) -> str:
    lines = code.strip().split("\n")
    max_len = max(len(line) for line in lines)
    width = max(max_len + 4, 40)

    result = []
    result.append("+" + "-" * (width - 2) + "+")
    for line in lines:
        result.append("| " + line.ljust(width - 4) + " |")
    result.append("+" + "-" * (width - 2) + "+")

    return "\n".join(result)


def main():
    from training.inference import generate_mermaid, load_model

    print_welcome()

    print("Loading model", end="", flush=True)
    for _ in range(3):
        print(".", end="", flush=True)
    print()

    try:
        model, tokenizer = load_model()
        print("Model loaded!\n")
    except Exception as e:
        print(f"\nFailed to load model: {e}")
        print("Make sure you've trained the model first: python -m training.train")
        sys.exit(1)

    while True:
        print(">>> ", end="", flush=True)
        user_input = read_multiline_input()

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.lower().strip()
            if cmd in ("/quit", "/exit", "/q"):
                print("\nGoodbye!")
                break
            elif cmd == "/help":
                print_help()
                continue
            else:
                print(f"Unknown command: {user_input}")
                print("Type /help for available commands.\n")
                continue

        print("\nGenerating...\n")
        try:
            mermaid_code = generate_mermaid(user_input, model, tokenizer)
        except Exception as e:
            print(f"Error: {e}\n")
            continue

        print(format_mermaid(mermaid_code))
        print()


if __name__ == "__main__":
    main()
