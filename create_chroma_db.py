import glob
import json
import os
import warnings
from typing import List

# Tắt cảnh báo LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
from huggingface_hub import login
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm

from utils import load_config


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


def main():
    print("=== BẮT ĐẦU TẠO CƠ SỞ DỮ LIỆU CHROMA ===")

    # Tải cấu hình
    config = load_config()
    if not config:
        print("Lỗi: Không thể đọc file config.yaml")
        return

    # Lấy thông số từ config
    db_path = config["db_dir"]
    embeddings_model = config["api"]["huggingface"]["embedding_model"]
    huggingface_token = config["api"]["huggingface"]["token"]
    data_path = config["data_path"]
    data_prefix = config["data_prefix"]

    # Cấu hình thiết bị
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Sử dụng thiết bị: {device}")

    # Đăng nhập HuggingFace nếu có token
    if huggingface_token:
        login(token=huggingface_token)
        print("Đã đăng nhập với token HuggingFace")

    # Tìm thư mục dữ liệu
    data_dirs = get_data_directories(data_path, data_prefix)
    if not data_dirs:
        print(
            f"Không tìm thấy thư mục nào bắt đầu bằng '{data_prefix}' trong '{data_path}'"
        )
        return

    print(
        f"Đã tìm thấy {len(data_dirs)} thư mục dữ liệu: {[os.path.basename(d) for d in data_dirs]}"
    )

    # Tạo thư mục db nếu chưa tồn tại
    os.makedirs(db_path, exist_ok=True)

    # --------------------------------------------------
    # BƯỚC 1: TẢI DỮ LIỆU
    # --------------------------------------------------
    print("\nBƯỚC 1: TẢI DỮ LIỆU TỪ THƯ MỤC SEGMENTED")
    documents = []

    for data_dir in data_dirs:
        segmented_dir = os.path.join(data_dir, "segmented")
        if not os.path.exists(segmented_dir):
            continue

        # Tìm tất cả file JSON
        json_files = glob.glob(
            os.path.join(segmented_dir, "**", "*.json"), recursive=True
        )
        if not json_files:
            print(f"Không tìm thấy file JSON nào trong {segmented_dir}")
            continue

        print(
            f"Đang tải dữ liệu từ {segmented_dir}... (tìm thấy {len(json_files)} file)"
        )

        # Xử lý từng file JSON
        dir_docs = []
        for json_file in tqdm(json_files, desc=f"Đọc file từ {segmented_dir}"):
            dir_docs.extend(load_json_documents(json_file))

        print(f"Đã tải {len(dir_docs)} đoạn văn bản từ {segmented_dir}")
        documents.extend(dir_docs)

    # Kiểm tra số lượng tài liệu
    if not documents:
        print("CẢNH BÁO: Không tìm thấy tài liệu nào!")
        return

    print(f"Đã tải tổng cộng {len(documents)} đoạn văn bản")

    # --------------------------------------------------
    # BƯỚC 2: XỬ LÝ DATABASE
    # --------------------------------------------------
    print("\nBƯỚC 2: TẠO HOẶC TẢI VECTOR DATABASE")

    # Kiểm tra vector database đã tồn tại chưa
    db_exists = os.path.exists(os.path.join(db_path, "chroma.sqlite3"))

    if not db_exists:
        # Tạo mới vector database
        create_vector_db(documents, db_path, embeddings_model, device)
    else:
        # Xác nhận từ người dùng
        print("Vector database đã tồn tại.")
        choice = input("Bạn có muốn xóa và tạo lại không? (y/n): ")

        if choice.lower() == "y":
            # Xóa và tạo lại database
            import shutil

            shutil.rmtree(db_path)
            os.makedirs(db_path, exist_ok=True)
            create_vector_db(documents, db_path, embeddings_model, device)
        else:
            # Tải database hiện có
            print("Đang tải vector database có sẵn...")
            embeddings = HuggingFaceEmbeddings(
                model_name=embeddings_model, model_kwargs={"device": device}
            )
            Chroma(persist_directory=db_path, embedding_function=embeddings)
            print("Đã tải vector database thành công")

    print("\n=== HOÀN THÀNH TẠO CƠ SỞ DỮ LIỆU CHROMA ===")
    print(f"Đường dẫn cơ sở dữ liệu: {db_path}")


if __name__ == "__main__":
    main()
