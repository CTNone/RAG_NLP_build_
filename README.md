# RAG Chatbot về Pittsburgh/CMU và VNU

Chatbot sử dụng kỹ thuật Retrieval Augmented Generation (RAG) để trả lời câu hỏi về Pittsburgh / Carnegie Mellon University (CMU) và VNU. Hệ thống sử dụng:

- Together AI API cho Large Language Model
- Hugging Face cho Embedding Model
- ChromaDB làm vector database

## Yêu cầu

- Python 3.8+
- CUDA (tùy chọn, để tăng tốc embedding)
- Token từ Together AI và Hugging Face (hướng dẫn bên dưới)

## Đăng ký và lấy API Token

### 1. Together AI

1. Truy cập [Together AI](https://api.together.ai/signin)
2. Đăng ký tài khoản mới
3. Vào [Dashboard](https://api.together.xyz/settings/api-keys)
4. Tạo API key mới và lưu lại

### 2. Hugging Face

1. Truy cập [Hugging Face](https://huggingface.co/join)
2. Đăng ký tài khoản mới
3. Vào [Settings > Access Tokens](https://huggingface.co/settings/tokens)
4. Tạo token mới với quyền "read"

## Cài đặt

1. Clone repository:

```bash
git clone <repository-url>
```

2. Cài đặt các thư viện:

```bash
pip install -r requirements.txt
```

3. Tùy chỉnh nội dung file `config.yaml` sao cho phù hợp:

```yaml
# API Credentials
api:
  huggingface:
    token: "your-huggingface-token" # Token từ Hugging Face
    embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  together:
    token: "your-together-token" # Token từ Together AI
    model_id: "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

# Database
db_dir: "chroma_db"

# Data Paths
data_path: "data" # Folder chứa các dữ liệu cần thiết

data_prefix: "DATA_" # Prefix để tự động tìm các thư mục dữ liệu

# Test Questions and Answers Paths
test_questions_dir: "data/test_questions"
test_answers_dir: "data/test_answers"

# Model Parameters
max_new_tokens: Số lượng token tối đa mà model có thể sinh ra trong câu trả lời
temperature: Độ ngẫu nhiên trong câu trả lời (0.0-1.0, càng thấp càng ổn định)
top_p: Ngưỡng xác suất tích lũy để chọn token tiếp theo (0.0-1.0)
repetition_penalty: Hệ số phạt cho việc lặp lại từ/cụm từ (>1.0 để giảm lặp lại)

# RAG Settings
retriever_k: Số lượng đoạn văn bản truy xuất cho mỗi câu hỏi

# Prompt Template
template:
```

## Chuẩn bị dữ liệu

### 1. Cấu trúc thư mục

```
data/
├── DATA_CMU_Pit/
│   └── segmented/
│       ├── about_cmu.json
│       ├── campus_life.json
│       └── ...
├── ...

```

### 2. Cấu trúc file JSON

Mỗi file JSON trong thư mục `segmented/` phải có cấu trúc sau:

```json
{
  "url": "https://example.com/source-page",
  "title": "Tiêu đề trang",
  "chunks": [
    "Đoạn văn bản thứ nhất...",
    "Đoạn văn bản thứ hai...",
    "Đoạn văn bản thứ ba..."
  ]
}
```

### 3. Tạo vector database

```bash
python create_chroma_db.py
```

## Chạy Chatbot

1. Khởi động ứng dụng:

```bash
python app.py
```

2. Truy cập giao diện web:

- Mở trình duyệt và truy cập địa chỉ hiển thị trong terminal (thường là http://127.0.0.1:7860)
- Bắt đầu đặt câu hỏi về Pittsburgh / CMU và VNU!

## Tài liệu

1. [Llama-3.3-70B Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
2. ...
