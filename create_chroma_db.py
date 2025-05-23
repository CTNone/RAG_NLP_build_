import glob
import os
import warnings
from typing import List

# Tắt cảnh báo của TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all messages, 3 = ERROR

import torch
from huggingface_hub import login
from tqdm import tqdm

from src import create_vector_db, load_config
from src.retriever import get_data_directories, load_json_documents


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
            print("Đang giữ nguyên vector database có sẵn...")

    print("\n=== HOÀN THÀNH TẠO CƠ SỞ DỮ LIỆU CHROMA ===")
    print(f"Đường dẫn cơ sở dữ liệu: {db_path}")


if __name__ == "__main__":
    main()
