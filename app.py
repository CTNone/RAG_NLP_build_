import os
import traceback
from typing import List, Optional, Tuple

import gradio as gr
import torch
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from together import Together

from utils import load_config

# Biến toàn cục
app_config = None
app_db = None


def load_chroma_db(config: dict) -> Chroma:
    """Tải Chroma database

    Args:
        config: Cấu hình ứng dụng

    Returns:
        Chroma database đã tải
    """
    db_path = config["db_dir"]
    embeddings_model = config["api"]["huggingface"]["embedding_model"]

    # Cấu hình thiết bị
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Khởi tạo embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=embeddings_model, model_kwargs={"device": device}
    )

    print(f"Đang tải Chroma database từ {db_path}...")
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    print(f"Đã tải thành công")
    return db


def retrieve_documents(query: str, k: int = 3) -> List[str]:
    """Truy xuất các đoạn văn bản liên quan

    Args:
        query: Câu hỏi người dùng
        k: Số lượng đoạn văn bản cần truy xuất

    Returns:
        Danh sách các đoạn văn bản liên quan
    """
    global app_db

    retriever = app_db.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)

    contexts = []
    for i, doc in enumerate(docs):
        contexts.append(f"[{i+1}] {doc.page_content}")

    print(f"Đã tìm thấy {len(docs)} đoạn văn bản liên quan")
    return contexts


def create_prompt(question: str, contexts: List[str]) -> str:
    """Tạo prompt từ câu hỏi và context

    Args:
        question: Câu hỏi người dùng
        contexts: Danh sách các đoạn văn bản liên quan

    Returns:
        Prompt hoàn chỉnh
    """
    global app_config

    context_text = "\n\n".join(contexts)
    prompt_template = PromptTemplate.from_template(app_config["template"])
    return prompt_template.format(context=context_text, question=question)


def call_llm_api(prompt: str, history: Optional[List[Tuple[str, str]]] = None) -> str:
    """Gọi API LLM để lấy phản hồi

    Args:
        prompt: Prompt hoàn chỉnh
        history: Lịch sử chat (tùy chọn)

    Returns:
        Phản hồi từ LLM
    """
    global app_config

    together_api_key = app_config["api"]["together"]["token"]
    model_id = app_config["api"]["together"]["model_id"]

    # Khởi tạo client
    os.environ["TOGETHER_API_KEY"] = together_api_key
    client = Together()

    # Chuẩn bị messages từ lịch sử chat
    messages = []
    if history and len(history) > 0:
        for user_msg, assistant_msg in history:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})

    # Thêm prompt hiện tại
    messages.append({"role": "user", "content": prompt})

    # Gọi API
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=app_config["temperature"],
        top_p=app_config["top_p"],
        max_tokens=app_config["max_new_tokens"],
    )

    if response and hasattr(response, "choices") and len(response.choices) > 0:
        return response.choices[0].message.content
    else:
        raise ValueError("Không nhận được phản hồi hợp lệ từ API")


def get_response(message: str, history=None) -> str:
    """Xử lý câu hỏi và tạo câu trả lời với RAG

    Args:
        message: Câu hỏi người dùng
        history: Lịch sử chat (tùy chọn)

    Returns:
        Câu trả lời
    """
    global app_config, app_db

    try:
        # Kiểm tra tài nguyên đã được tải
        if not app_config or not app_db:
            return "Lỗi: Tài nguyên chưa được tải. Hãy khởi động lại ứng dụng."

        # Lấy các tham số cấu hình
        retriever_k = app_config["retriever_k"]

        # Thực hiện truy vấn RAG
        print(f"Truy vấn: '{message}'")
        contexts = retrieve_documents(message, retriever_k)
        prompt = create_prompt(message, contexts)
        response = call_llm_api(prompt, history)

        return response

    except Exception as e:
        error_msg = f"Lỗi: {str(e)}\n"
        print(error_msg)
        print(traceback.format_exc())
        return f"Xin lỗi, có lỗi xảy ra: {str(e)}"


def initialize_app() -> bool:
    """Khởi tạo ứng dụng và tải tài nguyên

    Returns:
        True nếu khởi tạo thành công, False nếu thất bại
    """
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
        print(traceback.format_exc())
        return False


# Tạo giao diện Gradio
demo = gr.ChatInterface(
    fn=get_response,
    title="RAG Chatbot về Pittsburgh / CMU và VNU",
    description="Chatbot có khả năng truy vấn dữ liệu về Pittsburgh / CMU và VNU",
    examples=[
        "What bank, which is the 5th largest in the US, is based in Pittsburgh?",
        "How many neighborhoods does Pittsburgh have?",
        "Who is Pittsburgh named after?",
        "What famous vaccine was developed at University of Pittsburgh in 1955?",
        "DHGQHN được thành lập khi nào?",
        "VNU-USSH được thành lập khi nào",
    ],
    theme="soft",
)


if __name__ == "__main__":
    print("Đang khởi động RAG chatbot...")

    if initialize_app():
        demo.launch()
    else:
        print("Không thể khởi động ứng dụng do lỗi khởi tạo")
        exit(1)
