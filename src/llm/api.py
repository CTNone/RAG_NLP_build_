import os
from typing import Any, List, Optional, Tuple

from langchain.prompts import PromptTemplate
from together import Together


def create_prompt(question: str, contexts: List[str], template: str) -> str:
    """Tạo prompt từ câu hỏi và context"""
    context_text = "\n\n".join(contexts)
    prompt_template = PromptTemplate.from_template(template)
    return prompt_template.format(context=context_text, question=question)


def call_llm_api(prompt: str, config: dict, history: Optional[List[Any]] = None) -> str:
    """Gọi API LLM để lấy phản hồi"""
    # Thiết lập API
    os.environ["TOGETHER_API_KEY"] = config["api"]["together"]["token"]
    client = Together()
    model_id = config["api"]["together"]["model_id"]

    # Chuẩn bị messages từ lịch sử chat
    messages = []
    if history and len(history) > 0:
        for item in history:
            # Xử lý linh hoạt với lịch sử chat
            if isinstance(item, tuple) and len(item) == 2:
                user_msg, assistant_msg = item
                if user_msg:
                    messages.append({"role": "user", "content": user_msg})
                if assistant_msg:
                    messages.append({"role": "assistant", "content": assistant_msg})
            elif isinstance(item, list) and len(item) >= 2:
                # Trường hợp lịch sử được lưu dưới dạng list
                user_msg = item[0]
                assistant_msg = item[1] if len(item) > 1 else None
                if user_msg:
                    messages.append({"role": "user", "content": user_msg})
                if assistant_msg:
                    messages.append({"role": "assistant", "content": assistant_msg})
            elif isinstance(item, dict) and "user" in item and "assistant" in item:
                # Trường hợp lịch sử được lưu dưới dạng dict
                if item["user"]:
                    messages.append({"role": "user", "content": item["user"]})
                if item["assistant"]:
                    messages.append({"role": "assistant", "content": item["assistant"]})
            elif isinstance(item, str):
                # Trường hợp lịch sử chỉ là chuỗi
                messages.append({"role": "user", "content": item})

    # Thêm prompt hiện tại
    messages.append({"role": "user", "content": prompt})

    # Gọi API
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=config["temperature"],
        top_p=config["top_p"],
        max_tokens=config["max_new_tokens"],
    )

    if response and hasattr(response, "choices") and len(response.choices) > 0:
        return response.choices[0].message.content
    else:
        raise ValueError("Không nhận được phản hồi hợp lệ từ API")
