import re
from typing import Dict, Optional, Tuple


class OpenRAGTokens:
    IRRELEVANT = "[Irrelevant]"
    RELEVANT = "[Relevant]"

    NO_RETRIEVAL = "[No Retrieval]"
    RETRIEVAL = "[Retrieval]"
    CONTINUE_EVIDENCE = "[Continue to Use Evidence]"

    UTILITY_1 = "[Utility:1]"
    UTILITY_2 = "[Utility:2]"
    UTILITY_3 = "[Utility:3]"
    UTILITY_4 = "[Utility:4]"
    UTILITY_5 = "[Utility:5]"

    FULLY_SUPPORTED = "[Fully supported]"
    PARTIALLY_SUPPORTED = "[Partially supported]"
    NO_SUPPORT = "[No support / Contradictory]"

    PARAGRAPH_START = "<paragraph>"
    PARAGRAPH_END = "</paragraph>"

    END = "</s>"

    @classmethod
    def control_tokens(cls):
        return [
            cls.FULLY_SUPPORTED,
            cls.PARTIALLY_SUPPORTED,
            cls.NO_SUPPORT,
            cls.NO_RETRIEVAL,
            cls.RETRIEVAL,
            cls.IRRELEVANT,
            cls.RELEVANT,
            cls.PARAGRAPH_START,
            cls.PARAGRAPH_END,
            cls.UTILITY_1,
            cls.UTILITY_2,
            cls.UTILITY_3,
            cls.UTILITY_4,
            cls.UTILITY_5,
        ]


def _safe_token_id(tokenizer, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None and token_id == unk_id:
        ids = tokenizer(token, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            return int(ids[0])
    return int(token_id)


def load_special_tokens(tokenizer, use_groundness: bool = False, use_utility: bool = False):
    ret_tokens = {
        OpenRAGTokens.RETRIEVAL: _safe_token_id(tokenizer, OpenRAGTokens.RETRIEVAL),
        OpenRAGTokens.NO_RETRIEVAL: _safe_token_id(tokenizer, OpenRAGTokens.NO_RETRIEVAL),
        OpenRAGTokens.CONTINUE_EVIDENCE: _safe_token_id(tokenizer, OpenRAGTokens.CONTINUE_EVIDENCE),
    }

    rel_tokens = {
        OpenRAGTokens.RELEVANT: _safe_token_id(tokenizer, OpenRAGTokens.RELEVANT),
        OpenRAGTokens.IRRELEVANT: _safe_token_id(tokenizer, OpenRAGTokens.IRRELEVANT),
    }

    grd_tokens = None
    if use_groundness:
        grd_tokens = {
            OpenRAGTokens.FULLY_SUPPORTED: _safe_token_id(tokenizer, OpenRAGTokens.FULLY_SUPPORTED),
            OpenRAGTokens.PARTIALLY_SUPPORTED: _safe_token_id(tokenizer, OpenRAGTokens.PARTIALLY_SUPPORTED),
            OpenRAGTokens.NO_SUPPORT: _safe_token_id(tokenizer, OpenRAGTokens.NO_SUPPORT),
        }

    ut_tokens = None
    if use_utility:
        ut_tokens = {
            OpenRAGTokens.UTILITY_1: _safe_token_id(tokenizer, OpenRAGTokens.UTILITY_1),
            OpenRAGTokens.UTILITY_2: _safe_token_id(tokenizer, OpenRAGTokens.UTILITY_2),
            OpenRAGTokens.UTILITY_3: _safe_token_id(tokenizer, OpenRAGTokens.UTILITY_3),
            OpenRAGTokens.UTILITY_4: _safe_token_id(tokenizer, OpenRAGTokens.UTILITY_4),
            OpenRAGTokens.UTILITY_5: _safe_token_id(tokenizer, OpenRAGTokens.UTILITY_5),
        }

    return ret_tokens, rel_tokens, grd_tokens, ut_tokens


def postprocess_answer_option_conditioned(text: str) -> str:
    if text is None:
        return ""
    for token in OpenRAGTokens.control_tokens():
        text = text.replace(token, "")
    text = text.replace(OpenRAGTokens.END, "")
    text = text.replace("<|endoftext|>", "")
    text = text.replace("\n", " ")
    return " ".join(text.split()).strip()


def get_scoring_components(response: str) -> Dict[str, float]:
    relevance_score = 1.0 if OpenRAGTokens.RELEVANT in response else 0.0

    if OpenRAGTokens.FULLY_SUPPORTED in response:
        ground_score = 1.0
    elif OpenRAGTokens.PARTIALLY_SUPPORTED in response:
        ground_score = 0.5
    else:
        ground_score = 0.0

    utility_score = 0.0
    m = re.search(r"\[Utility:(\d+)\]", response or "")
    if m:
        raw = int(m.group(1))
        raw = max(1, min(5, raw))
        utility_score = (raw - 3) * 0.5

    return {
        "relevance_score": relevance_score,
        "ground_score": ground_score,
        "utility_score": utility_score,
    }
