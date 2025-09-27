import re
import json
from typing import Tuple


def extract_assistant_output(text: str) -> str:
    pattern = r"assistant\n\n(.*)$"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        output = match.group(1).strip()
        output = re.sub(r"\n\s*\n", "\n", output)
        return output
    return text


def extract_final_answer_and_rationale(
    text: str, question_type: str = "OEQ"
) -> Tuple[str, str]:
    if question_type == "MATH":
        # naive: last number
        nums = re.findall(r"[\d,]+(?:\.\d+)?", text)
        return (nums[-1].replace(",", "") if nums else "Error: No number found", text)
    pattern = r"Answer:\s*(.*?)\s*Rationale:\s*(.*)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "Answer not found", "Rationale not found"


def parse_query(response: str) -> str:
    # try JSON with search_query
    search_query_pattern = re.compile(
        r'\{\s*"search_query"\s*:\s*".*?"\s*\}', re.DOTALL
    )
    match = search_query_pattern.search(response)
    if match:
        try:
            js = json.loads(match.group(0))
            return js.get("search_query", "").strip()
        except Exception:
            pass
    salvage_pattern = re.compile(
        r"search_query.*?:(.*?)[,\n\r]+.*?reasoning", re.DOTALL
    )
    salvage_match = salvage_pattern.search(response)
    if salvage_match:
        return salvage_match.group(1).strip().strip('"')
    return response.strip()
