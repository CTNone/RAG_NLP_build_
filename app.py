import os
from typing import List, Optional, Tuple

import gradio as gr

from src import load_chroma_db, load_config, retrieve_documents
from src.llm import call_llm_api, create_prompt

# Biến toàn cục
app_config = None
app_db = None


def get_response(message: str, history=None) -> str:
    """Xử lý câu hỏi và tạo câu trả lời với RAG"""
    global app_config, app_db

    try:
        # Kiểm tra tài nguyên đã được tải
        if not app_config or not app_db:
            return "Lỗi: Tài nguyên chưa được tải. Hãy khởi động lại ứng dụng."

        # Thực hiện truy vấn RAG
        print(f"Truy vấn: '{message}'")
        contexts = retrieve_documents(app_db, message, app_config["retriever_k"])
        prompt = create_prompt(message, contexts, app_config["template"])
        return call_llm_api(prompt, app_config, history)

    except Exception as e:
        print(f"Lỗi: {str(e)}")
        return f"Xin lỗi, có lỗi xảy ra: {str(e)}"


def initialize_app() -> bool:
    """Khởi tạo ứng dụng và tải tài nguyên"""
    global app_config, app_db

    try:
        # Tải config
        app_config = load_config()
        if not app_config:
            print("Lỗi: Không thể đọc file config.yaml")
            return False

        print("Đã tải config:")
        print(f"- Model: {app_config['api']['together']['model_id']}")
        print(f"- Embedding: {app_config['api']['huggingface']['embedding_model']}")

        # Tải Chroma database
        app_db = load_chroma_db(app_config)
        return True

    except Exception as e:
        print(f"Lỗi khởi tạo: {str(e)}")
        return False


# Khởi tạo giao diện Gradio
examples = [
    "What bank, which is the 5th largest in the US, is based in Pittsburgh?",
    "How many neighborhoods does Pittsburgh have?",
    "Who is Pittsburgh named after?",
    "What famous vaccine was developed at University of Pittsburgh in 1955?",
    "DHGQHN được thành lập khi nào?",
    "VNU-USSH được thành lập khi nào",
]

# Sử dụng ChatInterface mà không có tham số message_type
demo = gr.ChatInterface(
    fn=get_response,
    title="RAG Chatbot về Pittsburgh / CMU và VNU",
    description="Chatbot có khả năng truy vấn dữ liệu về Pittsburgh / CMU và VNU",
    examples=examples,
    theme="soft",
    type="messages",  # Sử dụng type thay vì message_type
)


if __name__ == "__main__":
    print("Đang khởi động RAG chatbot...")

    if initialize_app():
        demo.launch()
    else:
        print("Không thể khởi động ứng dụng do lỗi khởi tạo")
        exit(1)
