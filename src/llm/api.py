import json
import os
import urllib.error
import urllib.request
from typing import Any, List, Optional

# from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate

try:
    from together import Together
except ImportError:
    Together = None


def create_prompt(question: str, contexts: List[str], template: str) -> str:
    """Tạo prompt từ câu hỏi và context"""
    context_text = "".join(contexts)
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
        raise RuntimeError(f"OpenRouter request failed: {exc.code} {exc.reason} - {error_body}")
    result = json.loads(raw)
    if not result or "choices" not in result or len(result["choices"]) == 0:
        raise ValueError("Không nhận được phản hồi hợp lệ từ OpenRouter")
    return result["choices"][0]["message"]["content"]


def _call_together(prompt: str, config: dict, history: Optional[List[Any]] = None) -> str:
    if Together is None:
        raise ImportError("Library 'together' is required for Together API provider")
    os.environ["TOGETHER_API_KEY"] = config["api"]["together"]["token"]
    client = Together()
    model_id = config["api"]["together"]["model_id"]
    messages = _prepare_messages(history, prompt)
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=config["temperature"],
        top_p=config["top_p"],
        max_tokens=config["max_new_tokens"],
    )
    if response and hasattr(response, "choices") and len(response.choices) > 0:
        return response.choices[0].message.content
    raise ValueError("Không nhận được phản hồi hợp lệ từ Together API")


def call_llm_api(prompt: str, config: dict, history: Optional[List[Any]] = None) -> str:
    provider = config["api"].get("provider", "together").lower()
    if provider == "openrouter":
        return _call_openrouter(prompt, config, history)
    if provider == "together":
        return _call_together(prompt, config, history)
    raise ValueError(f"Unknown API provider: {provider}")
