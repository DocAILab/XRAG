"""Generic Hugging Face causal-chat model adapter."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from threading import Thread
from typing import Any, Generator, Optional

import torch
from llama_index.core.llms import CompletionResponse, CustomLLM, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from pydantic import Field
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from .huggingface_templates import TEMPLATE_NAMES, messages_to_prompt, stop_token_ids


LEGACY_MODEL_ALIASES = {
    "llama": "meta-llama/Llama-2-7b-chat-hf",
    "chatglm": "THUDM/chatglm3-6b",
    "qwen": "Qwen/Qwen1.5-7B-Chat",
    "qwen14_int8": "Qwen/Qwen1.5-14B-Chat-GPTQ-Int8",
    "qwen7_int8": "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8",
    "qwen1.8": "Qwen/Qwen1.5-1.8B-Chat",
    "baichuan": "baichuan-inc/Baichuan2-7B-Chat",
    "falcon": "tiiuae/falcon-7b-instruct",
    "mpt": "mosaicml/mpt-7b-chat",
    "yi": "01-ai/Yi-6B-Chat",
}


class TransformersChatLLM(CustomLLM):
    """LlamaIndex adapter backed directly by a Transformers causal LM."""

    model: Any = Field(exclude=True)
    tokenizer: Any = Field(exclude=True)
    model_name: str
    context_window: int
    max_new_tokens: int
    stopping_ids: list[int] = Field(default_factory=list)
    generate_kwargs: dict[str, Any] = Field(default_factory=dict)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_new_tokens,
            model_name=self.model_name,
            is_chat_model=True,
        )

    def _format_prompt(self, prompt: str, formatted: bool) -> str:
        if formatted:
            return prompt
        if self.query_wrapper_prompt:
            prompt = self.query_wrapper_prompt.format(query_str=prompt)
        if self.completion_to_prompt:
            return self.completion_to_prompt(prompt)
        return f"{self.system_prompt}\n\n{prompt}" if self.system_prompt else prompt

    def _model_inputs(self, prompt: str) -> Any:
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        device = getattr(self.model, "device", None)
        return inputs.to(device) if device is not None else inputs

    def _generation_options(self, **kwargs: Any) -> dict[str, Any]:
        options = {**self.generate_kwargs, **kwargs}
        options.setdefault("max_new_tokens", self.max_new_tokens)
        if self.stopping_ids:
            options.setdefault("eos_token_id", self.stopping_ids)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            options.setdefault("pad_token_id", pad_id)
        return options

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        prompt = self._format_prompt(prompt, formatted)
        inputs = self._model_inputs(prompt)
        output = self.model.generate(**inputs, **self._generation_options(**kwargs))
        input_length = inputs["input_ids"].shape[-1]
        text = self.tokenizer.decode(output[0][input_length:], skip_special_tokens=True)
        return CompletionResponse(text=text, raw=output)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> Generator[CompletionResponse, None, None]:
        prompt = self._format_prompt(prompt, formatted)
        inputs = self._model_inputs(prompt)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        errors: list[BaseException] = []

        def generate() -> None:
            try:
                self.model.generate(
                    **inputs,
                    streamer=streamer,
                    **self._generation_options(**kwargs),
                )
            except BaseException as exc:
                errors.append(exc)
                streamer.on_finalized_text("", stream_end=True)

        worker = Thread(target=generate, daemon=True)
        worker.start()
        full_text = ""
        for delta in streamer:
            full_text += delta
            yield CompletionResponse(text=full_text, delta=delta)
        worker.join()
        if errors:
            raise errors[0]


def resolve_model_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise ValueError("A Hugging Face repository ID or local model path is required.")
    return LEGACY_MODEL_ALIASES.get(name, name)


def _context_window(model_config: Any, requested: int) -> int:
    candidates = []
    for field in ("max_position_embeddings", "n_positions", "seq_length"):
        value = getattr(model_config, field, None)
        if isinstance(value, int) and 0 < value < 10_000_000:
            candidates.append(value)
    detected = min(candidates) if candidates else 4096
    return min(requested, detected) if requested > 0 else detected


def _dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def get_huggingfacellm(
    name: str,
    *,
    auth_token: Optional[str] = None,
    template: str = "auto",
    context_window: int = 0,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    trust_remote_code: bool = False,
) -> TransformersChatLLM:
    """Load any Transformers causal LM with native or explicit chat formatting.

    Without an explicit token, use the login saved by the Hugging Face CLI.
    Read the credential file directly so environment credentials are not used.
    """
    if template not in TEMPLATE_NAMES:
        raise ValueError(f"Unknown Hugging Face chat template: {template}")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")

    model_name = resolve_model_name(name)
    token: Any = auth_token.strip() if auth_token and auth_token.strip() else False
    if token is False:
        from huggingface_hub.constants import HF_TOKEN_PATH

        try:
            token = Path(HF_TOKEN_PATH).read_text(encoding="utf-8").strip() or False
        except FileNotFoundError:
            pass
    shared = {"token": token, "trust_remote_code": bool(trust_remote_code)}
    config = AutoConfig.from_pretrained(model_name, **shared)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **shared)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    resolved_context = _context_window(config, int(context_window or 0))
    if max_new_tokens >= resolved_context:
        raise ValueError("Maximum generated tokens must be smaller than the context window.")
    renderer = partial(messages_to_prompt, tokenizer, template)
    # Validate the selected format before downloading or allocating model weights.
    renderer([{"role": "user", "content": "template validation"}])

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=_dtype(),
        **shared,
    )
    generate_kwargs = {"do_sample": temperature > 0}
    if temperature > 0:
        generate_kwargs["temperature"] = float(temperature)

    return TransformersChatLLM(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        context_window=resolved_context,
        max_new_tokens=max_new_tokens,
        messages_to_prompt=renderer,
        completion_to_prompt=lambda prompt: renderer([{"role": "user", "content": prompt}]),
        stopping_ids=stop_token_ids(tokenizer, template),
        generate_kwargs=generate_kwargs,
    )
