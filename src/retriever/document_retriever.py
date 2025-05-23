import json
import os
from typing import List

from langchain.schema import Document


def retrieve_documents(db, query: str, k: int = 3) -> List[str]:
    """Truy xuất các đoạn văn bản liên quan"""
    retriever = db.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)

    contexts = [f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs)]
    return contexts


def load_json_documents(file_path: str) -> List[Document]:
    """Đọc file JSON segmented và chuyển đổi thành documents"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            if "chunks" not in data or not isinstance(data["chunks"], list):
                print(f"Cấu trúc JSON không đúng định dạng: {file_path}")
                return []

            docs = []
            url = data.get("url", "")
            title = data.get("title", "")

            for i, chunk in enumerate(data["chunks"]):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": file_path,
                            "url": url,
                            "title": title,
                            "chunk_id": i,
                        },
                    )
                )
            return docs
    except Exception as e:
        print(f"Lỗi khi đọc file JSON {file_path}: {str(e)}")
        return []


def get_data_directories(base_path: str, prefix: str) -> List[str]:
    """Tự động tìm các thư mục con bắt đầu bằng prefix"""
    if not os.path.exists(base_path):
        return []

    return [
        os.path.join(base_path, item)
        for item in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, item)) and item.startswith(prefix)
    ]
