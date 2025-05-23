from typing import List

import torch
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def load_chroma_db(config: dict) -> Chroma:
    """Tải Chroma database"""
    db_path = config["db_dir"]
    embeddings_model = config["api"]["huggingface"]["embedding_model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Khởi tạo embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=embeddings_model, model_kwargs={"device": device}
    )

    print(f"Đang tải Chroma database từ {db_path}...")
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    print(f"Đã tải thành công")
    return db


def create_vector_db(
    documents: List[Document], db_path: str, embeddings_model: str, device: str
) -> None:
    """Tạo và lưu vector database"""
    # Khởi tạo mô hình embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=embeddings_model, model_kwargs={"device": device}
    )

    # Tạo và lưu database
    Chroma.from_documents(
        documents=documents, embedding=embeddings, persist_directory=db_path
    )
    print(f"Đã tạo và lưu vector database với {len(documents)} đoạn văn bản")
