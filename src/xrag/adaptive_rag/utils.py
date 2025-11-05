from pathlib import Path


def load_prompt(prompt_name: str) -> str:
    """
    Load a prompt template from the adaptive_rag prompts directory.

    Args:
        prompt_name (str): Name of the prompt file (without .txt extension)

    Returns:
        The prompt template content

    Raises:
        FileNotFoundError: If the prompt file doesn't exist
    """
    prompts_dir = Path(__file__).parent.parent / "prompts" / "adaptive_rag"
    prompt_path = prompts_dir / f"{prompt_name}.txt"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()
