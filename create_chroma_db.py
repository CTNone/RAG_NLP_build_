import glob
import json
import os
import warnings
from typing import List

# Tắt cảnh báo LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

import torch
from huggingface_hub import login
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm

from utils import load_config


def load_json_documents(file_path: str) -> List[Document]:
    """Đọc các file JSON segmented và chuyển đổi thành documents

    Args:
        file_path: Đường dẫn đến file JSON

    Returns:
        Danh sách các Document
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            # Kiểm tra cấu trúc JSON: "url", "title", "chunks"
            if "chunks" in data and isinstance(data["chunks"], list):
                docs = []
                url = data.get("url", "")
                title = data.get("title", "")

                # Tạo một Document cho mỗi chunk
                for i, chunk in enumerate(data["chunks"]):
                    metadata = {
                        "source": file_path,
                        "url": url,
                        "title": title,
                        "chunk_id": i,
                    }
                    docs.append(Document(page_content=chunk, metadata=metadata))

                return docs
            else:
                print(f"Cấu trúc JSON không đúng định dạng: {file_path}")
                return []
    except Exception as e:
        print(f"Lỗi khi đọc file JSON {file_path}: {str(e)}")
        return []


def main():
    print("=== BẮT ĐẦU TẠO CƠ SỞ DỮ LIỆU CHROMA TỪ THƯ MỤC SEGMENTED ===")

    # Load cấu hình
    config = load_config()
    if not config:
        print("Lỗi: Không thể đọc file config.yaml")
        return

    # Lấy các thông số từ config
    db_path = config["db_dir"]
    embeddings_model = config["api"]["huggingface"]["embedding_model"]
    huggingface_token = config["api"]["huggingface"]["token"]
    data_path = config["data_path"]
    data_dirs = [os.path.join(data_path, subdir) for subdir in config["data_subdirs"]]

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(db_path, exist_ok=True)

    # Thiết lập token HuggingFace (nếu có)
    if huggingface_token:
        login(token=huggingface_token)
        print("Đã đăng nhập với token HuggingFace")
    else:
        print(
            "CẢNH BÁO: Không tìm thấy token HuggingFace. Một số chức năng có thể bị hạn chế."
        )

    # Cấu hình thiết bị
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Sử dụng thiết bị: {device}")

    # --------------------------------------------------
    # BƯỚC 1: TẢI DỮ LIỆU TỪ THƯ MỤC SEGMENTED
    # --------------------------------------------------
    print("\nBƯỚC 1: TẢI DỮ LIỆU TỪ THƯ MỤC SEGMENTED")
    documents = []

    for data_dir in data_dirs:
        segmented_dir = os.path.join(data_dir, "segmented")

        if os.path.exists(segmented_dir):
            print(f"Đang tải dữ liệu từ {segmented_dir}...")

            # Tìm tất cả các file JSON trong thư mục segmented
            json_files = glob.glob(
                os.path.join(segmented_dir, "**", "*.json"), recursive=True
            )

            if json_files:
                print(f"Tìm thấy {len(json_files)} file JSON.")

                for json_file in tqdm(json_files, desc=f"Đọc file từ {segmented_dir}"):
                    docs = load_json_documents(json_file)
                    documents.extend(docs)

                print(f"Đã tải {len(documents)} đoạn văn bản từ {segmented_dir}")
            else:
                print(f"Không tìm thấy file JSON nào trong {segmented_dir}")

    print(f"Đã tải tổng cộng {len(documents)} đoạn văn bản")

    if not documents:
        print(
            "CẢNH BÁO: Không tìm thấy tài liệu nào. Hãy kiểm tra lại các thư mục segmented."
        )
        return

    # --------------------------------------------------
    # BƯỚC 2: TẠO HOẶC TẢI VECTOR DATABASE
    # --------------------------------------------------
    print("\nBƯỚC 2: TẠO HOẶC TẢI VECTOR DATABASE")

    # Khởi tạo mô hình embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=embeddings_model, model_kwargs={"device": device}
    )

    # Kiểm tra xem đã có vector database hay chưa
    if not os.path.exists(os.path.join(db_path, "chroma.sqlite3")):
        # Nếu chưa có, tạo mới từ dữ liệu đã được phân đoạn
        print("Đang tạo vector database mới...")
        # Chroma tự động lưu khi được khởi tạo với persist_directory
        Chroma.from_documents(
            documents=documents, embedding=embeddings, persist_directory=db_path
        )
        print(f"Đã tạo và lưu vector database với {len(documents)} đoạn văn bản")
    else:
        # Nếu đã có, hỏi người dùng có muốn xóa và tạo lại không
        print("Vector database đã tồn tại.")
        choice = input("Bạn có muốn xóa và tạo lại không? (y/n): ")
        if choice.lower() == "y":
            import shutil

            # Xóa thư mục cũ
            shutil.rmtree(db_path)
            os.makedirs(db_path, exist_ok=True)

            # Tạo database mới
            print("Đang tạo vector database mới...")
            # Chroma tự động lưu khi được khởi tạo với persist_directory
            Chroma.from_documents(
                documents=documents, embedding=embeddings, persist_directory=db_path
            )
            print(f"Đã tạo và lưu vector database với {len(documents)} đoạn văn bản")
        else:
            # Tải database đã có
            print("Đang tải vector database có sẵn...")
            db = Chroma(persist_directory=db_path, embedding_function=embeddings)
            print("Đã tải vector database thành công")

    print("\n=== HOÀN THÀNH TẠO CƠ SỞ DỮ LIỆU CHROMA ===")
    print(f"Đường dẫn cơ sở dữ liệu: {db_path}")


if __name__ == "__main__":
    main()
