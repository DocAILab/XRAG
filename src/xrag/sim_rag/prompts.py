from typing import Dict
from pathlib import Path


_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts" / "sim_rag"


def _read_required(filename: str) -> str:
    path = _PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Required prompt file not found: {path}. Ensure templates exist under src/xrag/prompts/sim_rag."
        )
    return path.read_text(encoding="utf-8")


def _interpolate(template: str, variables: Dict[str, str]) -> str:
    out = template
    for k, v in variables.items():
        out = out.replace(f"{{{{{k}}}}}", v)
    return out


formatting = _read_required("formatting.json.txt")
formatting2 = _read_required("formatting_answer.txt")
formatting_abstain = _read_required("formatting_answer_or_unsure.txt")

force_answer_prompt = _interpolate(
    _read_required("force_answer_prompt.txt"),
    {"formatting_answer": formatting2},
)
abstain_force_answer_prompt = _interpolate(
    _read_required("abstain_force_answer_prompt.txt"),
    {"formatting_answer_or_unsure": formatting_abstain},
)
search_query_prompt = _interpolate(
    _read_required("search_query_prompt.txt"),
    {"formatting_json": formatting},
)
