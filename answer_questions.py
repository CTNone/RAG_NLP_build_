import csv
import os
import time
import warnings
from typing import Dict, List, Tuple

# Tắt cảnh báo của TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all messages, 3 = ERROR


import pandas as pd
from tqdm import tqdm

from src import load_chroma_db, load_config, retrieve_documents
from src.llm import call_llm_api, create_prompt


def setup_directories(config: Dict) -> None:
    """Tạo thư mục test_answers nếu chưa tồn tại"""
    os.makedirs(config["test_answers_dir"], exist_ok=True)


def load_questions_from_csv(file_path: str) -> List[str]:
    """Đọc câu hỏi từ file CSV"""
    questions = []

    try:
        df = pd.read_csv(file_path)
        # Lấy tên cột đầu tiên (thường là "Question" hoặc "Questions")
        question_column = df.columns[0]
        questions = df[question_column].tolist()
        print(f"Đã đọc {len(questions)} câu hỏi từ {file_path}")
    except Exception as e:
        print(f"Lỗi khi đọc file {file_path}: {str(e)}")

    return questions


def get_answer(question: str, db, config: Dict) -> str:
    """Lấy câu trả lời cho một câu hỏi"""
    try:
        # Truy xuất các đoạn văn bản liên quan
        contexts = retrieve_documents(db, question, config["retriever_k"])

        # Tạo prompt
        prompt = create_prompt(question, contexts, config["template"])

        # Gọi API LLM để lấy câu trả lời
        answer = call_llm_api(prompt, config, None)

        return answer
    except Exception as e:
        print(f"Lỗi khi xử lý câu hỏi '{question}': {str(e)}")
        return f"ERROR: {str(e)}"


def process_question_file(
    file_path: str, db, config: Dict
) -> Tuple[List[str], List[str]]:
    """Xử lý một file câu hỏi và trả về danh sách câu hỏi và câu trả lời"""
    questions = load_questions_from_csv(file_path)
    answers = []

    print(f"Đang xử lý {len(questions)} câu hỏi từ {os.path.basename(file_path)}...")

    for question in tqdm(questions):
        answer = get_answer(question, db, config)
        answers.append(answer)
        # Đợi một chút để tránh gọi API quá nhanh
        time.sleep(0.5)

    return questions, answers


def save_answers_to_csv(
    output_path: str, questions: List[str], answers: List[str]
) -> None:
    """Lưu câu hỏi và câu trả lời vào file CSV"""
    try:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Question", "Answer"])

            for q, a in zip(questions, answers):
                writer.writerow([q, a])

        print(f"Đã lưu kết quả vào {output_path}")
    except Exception as e:
        print(f"Lỗi khi lưu file {output_path}: {str(e)}")


def main():
    print("=== BẮT ĐẦU XỬ LÝ CÂU HỎI VÀ TẠO CÂU TRẢ LỜI ===")

    # Tải cấu hình
    config = load_config()
    if not config:
        print("Lỗi: Không thể đọc file config.yaml")
        return

    # Tạo thư mục đầu ra
    setup_directories(config)

    # Tải Chroma database
    db = load_chroma_db(config)

    # Lấy danh sách file câu hỏi
    questions_dir = config["test_questions_dir"]
    answers_dir = config["test_answers_dir"]

    csv_files = [f for f in os.listdir(questions_dir) if f.endswith(".csv")]

    if not csv_files:
        print(f"Không tìm thấy file CSV nào trong {questions_dir}")
        return

    print(f"Tìm thấy {len(csv_files)} file câu hỏi")

    # Xử lý từng file câu hỏi
    for csv_file in csv_files:
        input_path = os.path.join(questions_dir, csv_file)
        output_path = os.path.join(answers_dir, f"answers_{csv_file}")

        print(f"\nĐang xử lý file {csv_file}...")
        questions, answers = process_question_file(input_path, db, config)

        # Lưu kết quả
        save_answers_to_csv(output_path, questions, answers)

    print("\n=== HOÀN THÀNH XỬ LÝ CÂU HỎI VÀ TẠO CÂU TRẢ LỜI ===")


if __name__ == "__main__":
    main()
