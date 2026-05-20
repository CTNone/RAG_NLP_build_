import json
import os
import urllib.error
import urllib.request
from typing import Any, List, Optional

from langchain_core.prompts import PromptTemplate


def _format_context(context: Any, fallback_id: int) -> str:
    if not isinstance(context, dict):
        return str(context)

    citation_id = context.get("citation_id", fallback_id)
    title = context.get("title") or "Untitled source"
    section = context.get("section") or ""
    source = context.get("url") or context.get("source") or "unknown source"
    chunk_id = context.get("chunk_id", "")
    content = context.get("content", "")
    score = context.get("score")
    score_text = f" | score={score:.4f}" if isinstance(score, (int, float)) else ""

    header = f"[{citation_id}] {title}{score_text}"
    if section:
        header = f"{header} | {section}"

    return (
        f"{header}\n"
        f"Source: {source}\n"
        f"Chunk: {chunk_id}\n"
        f"Content:\n{content}"
    )


def _apply_context_budget(contexts: List[Any], config: Optional[dict]) -> List[Any]:
    config = config or {}
    ctx_cfg = config.get("context", {}) if isinstance(config.get("context", {}), dict) else {}
    max_chars = int(ctx_cfg.get("max_chars", 6000))
    max_chars_per_chunk = int(ctx_cfg.get("max_chars_per_chunk", 1500))

    pruned: List[Any] = []
    total = 0

    for context in contexts:
        if isinstance(context, dict):
            content = (context.get("content") or "").strip()
            if len(content) > max_chars_per_chunk:
                content = content[:max_chars_per_chunk]
            new_ctx = dict(context)
            new_ctx["content"] = content
        else:
            new_ctx = context

        rendered = _format_context(new_ctx, len(pruned) + 1)
        if pruned and total + len(rendered) > max_chars:
            break
        pruned.append(new_ctx)
        total += len(rendered)

    return pruned


def create_prompt(
    question: str,
    contexts: List[Any],
    template: str,
    config: Optional[dict] = None,
) -> str:
    """Create a prompt from question + retrieved contexts with citations."""
    contexts = _apply_context_budget(contexts, config)
    context_text = "\n\n---\n\n".join(
        _format_context(context, i + 1) for i, context in enumerate(contexts)
    )
    prompt_template = PromptTemplate.from_template(template)
    return prompt_template.format(context=context_text, question=question)


def _prepare_messages(history: Optional[List[Any]], prompt: str) -> list:
    messages = []
    if history:
        for item in history:
            if isinstance(item, tuple) and len(item) == 2:
                user_msg, assistant_msg = item
                if user_msg:
                    messages.append({"role": "user", "content": user_msg})
                if assistant_msg:
                    messages.append({"role": "assistant", "content": assistant_msg})
            elif isinstance(item, list) and len(item) >= 2:
                user_msg = item[0]
                assistant_msg = item[1] if len(item) > 1 else None
                if user_msg:
                    messages.append({"role": "user", "content": user_msg})
                if assistant_msg:
                    messages.append({"role": "assistant", "content": assistant_msg})
            elif isinstance(item, dict) and "user" in item and "assistant" in item:
                if item["user"]:
                    messages.append({"role": "user", "content": item["user"]})
                if item["assistant"]:
                    messages.append({"role": "assistant", "content": item["assistant"]})
            elif isinstance(item, str):
                messages.append({"role": "user", "content": item})

    messages.append({"role": "user", "content": prompt})
    return messages


def _call_openrouter(prompt: str, config: dict, history: Optional[List[Any]] = None) -> str:
    api_key = config["api"]["openrouter"]["token"]
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not configured")

    model_id = config["api"]["openrouter"]["model_id"]
    api_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost:7860",
        "X-Title": "RAG Chatbot",
    }
    messages = _prepare_messages(history, prompt)
    body = {
        "model": model_id,
        "messages": messages,
        "temperature": config.get("temperature", 0.2),
        "top_p": config.get("top_p", 0.92),
        "max_tokens": config.get("max_new_tokens", 192),
    }
    request = urllib.request.Request(
        api_url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8")
        raise RuntimeError(
            f"OpenRouter request failed: {exc.code} {exc.reason} - {error_body}"
        )
    result = json.loads(raw)
    if not result or "choices" not in result or len(result["choices"]) == 0:
        raise ValueError("No valid response from OpenRouter")
    return result["choices"][0]["message"]["content"]


def call_llm_api(
    prompt: str,
    config: dict,
    history: Optional[List[Any]] = None,
    stream: bool = False,
) -> Any:
    provider = config["api"].get("provider", "openrouter").lower()
    if provider == "openrouter":
        if stream:
            raise ValueError("Streaming is disabled in this build")
        return _call_openrouter(prompt, config, history)
    raise ValueError(f"Unknown or unsupported API provider: {provider}")

