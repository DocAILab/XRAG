# Copyright 2025 the LlamaFactory team.
# Adaptations Copyright 2026 DocAILab XRAG contributors.
# SPDX-License-Identifier: Apache-2.0
"""Small, dependency-free chat-template compatibility layer.

The fallback formats are adapted from the Apache-2.0 licensed LLaMA-Factory
template catalogue.  See ``THIRD_PARTY_NOTICES.md`` for attribution.  Native
tokenizer chat templates always take precedence; these formatters exist for
older model repositories that do not publish one.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Mapping


TEMPLATE_NAMES = ("auto", "chatml", "llama2", "llama3", "mistral", "gemma", "phi3", "plain")


def _normalise(messages: Iterable[Any]) -> list[dict[str, str]]:
    result = []
    for message in messages:
        role = getattr(message, "role", None)
        role = getattr(role, "value", role)
        content = getattr(message, "content", None)
        if isinstance(message, Mapping):
            role, content = message.get("role"), message.get("content")
        role = {"human": "user", "ai": "assistant"}.get(str(role), str(role))
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported chat role for Hugging Face model: {role!r}")
        if not isinstance(content, str):
            raise ValueError("Hugging Face chat messages must contain text content.")
        result.append({"role": role, "content": content})
    return result


def _render_tagged(messages, tags, prefix=""):
    if not messages or messages[-1]["role"] not in {"user", "tool"}:
        raise ValueError("The fallback chat template expects the final message to request an assistant response.")
    chunks = [prefix]
    for message in messages:
        start, end = tags[message["role"]]
        chunks.append(f"{start}{message['content']}{end}")
    chunks.append(tags["assistant"][0])
    return "".join(chunks)


def _chatml(messages):
    tags = {role: (f"<|im_start|>{role}\n", "<|im_end|>\n") for role in ("system", "user", "assistant", "tool")}
    return _render_tagged(messages, tags)


def _llama3(messages):
    tags = {role: (f"<|start_header_id|>{'ipython' if role == 'tool' else role}<|end_header_id|>\n\n", "<|eot_id|>") for role in ("system", "user", "assistant", "tool")}
    return _render_tagged(messages, tags, "<|begin_of_text|>")


def _phi3(messages):
    tags = {role: (f"<|{role}|>\n", "<|end|>\n") for role in ("system", "user", "assistant", "tool")}
    return _render_tagged(messages, tags)


def _paired(messages, style):
    system = []
    turns = []
    pending_user = None
    for message in messages:
        role, content = message["role"], message["content"]
        if role == "system" and not turns and pending_user is None:
            system.append(content)
        elif role == "tool":
            raise ValueError(f"The {style} fallback does not support tool messages.")
        elif role == "user":
            if pending_user is not None:
                raise ValueError(f"The {style} fallback requires alternating user and assistant turns.")
            pending_user = content
        elif role == "assistant":
            if pending_user is None:
                raise ValueError(f"The {style} fallback requires an assistant turn to follow a user turn.")
            turns.append((pending_user, content))
            pending_user = None
        else:
            raise ValueError(f"The {style} fallback only allows a leading system message.")
    if pending_user is None:
        raise ValueError(f"The {style} fallback expects the final message to be from the user.")
    return "\n\n".join(system), turns, pending_user


def _llama2(messages):
    system, turns, final_user = _paired(messages, "Llama 2")
    prompt = ""
    for index, (user, assistant) in enumerate(turns + [(final_user, None)]):
        if index == 0 and system:
            user = f"<<SYS>>\n{system}\n<</SYS>>\n\n{user}"
        prompt += f"<s>[INST] {user} [/INST]"
        if assistant is not None:
            prompt += f" {assistant} </s>"
    return prompt


def _mistral(messages):
    system, turns, final_user = _paired(messages, "Mistral")
    prompt = "<s>"
    for index, (user, assistant) in enumerate(turns + [(final_user, None)]):
        if index == 0 and system:
            user = f"{system}\n\n{user}"
        prompt += f"[INST] {user}[/INST]"
        if assistant is not None:
            prompt += f" {assistant}</s>"
    return prompt


def _gemma(messages):
    system, turns, final_user = _paired(messages, "Gemma")
    chunks = ["<bos>"]
    for index, (user, assistant) in enumerate(turns + [(final_user, None)]):
        if index == 0 and system:
            user = f"{system}\n\n{user}"
        chunks.append(f"<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n")
        if assistant is not None:
            chunks.append(f"{assistant}<end_of_turn>\n")
    return "".join(chunks)


def _plain(messages):
    if not messages or messages[-1]["role"] not in {"user", "tool"}:
        raise ValueError("The plain fallback expects the final message to request an assistant response.")
    labels = {"system": "System", "user": "Human", "assistant": "Assistant", "tool": "Tool"}
    return "\n".join(f"{labels[m['role']]}: {m['content']}" for m in messages) + "\nAssistant:"


_RENDERERS: Dict[str, Callable[[list[dict[str, str]]], str]] = {
    "chatml": _chatml,
    "llama2": _llama2,
    "llama3": _llama3,
    "mistral": _mistral,
    "gemma": _gemma,
    "phi3": _phi3,
    "plain": _plain,
}


def messages_to_prompt(tokenizer: Any, template: str, messages: Iterable[Any]) -> str:
    """Render messages using the tokenizer's template or an explicit fallback."""
    normalised = _normalise(messages)
    if template == "auto":
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError(
                "This tokenizer does not define a chat_template. Choose an explicit "
                "Hugging Face chat template in Step 3."
            )
        try:
            return tokenizer.apply_chat_template(normalised, tokenize=False, add_generation_prompt=True)
        except Exception as exc:
            raise ValueError(f"The model's native chat template rejected these messages: {exc}") from exc
    try:
        return _RENDERERS[template](normalised)
    except KeyError as exc:
        raise ValueError(f"Unknown Hugging Face chat template: {template}") from exc


def stop_token_ids(tokenizer: Any, template: str) -> list[int]:
    tokens = {
        "chatml": ("<|im_end|>",),
        "llama3": ("<|eot_id|>", "<|eom_id|>"),
        "gemma": ("<end_of_turn>",),
        "phi3": ("<|end|>",),
    }.get(template, ())
    result = []
    if getattr(tokenizer, "eos_token_id", None) is not None:
        result.append(int(tokenizer.eos_token_id))
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for token in tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != unk_id and token_id not in result:
            result.append(int(token_id))
    return result
