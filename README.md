# 🧠 RAG Chatbot — Pittsburgh / CMU / VNU

Chatbot sử dụng **Retrieval-Augmented Generation (RAG)** để trả lời câu hỏi về **Pittsburgh**, **Carnegie Mellon University (CMU)** và **Đại học Quốc gia Hà Nội (VNU)**.

Hệ thống kết hợp:
- **Vector Database (ChromaDB)** — lưu trữ và tìm kiếm ngữ cảnh từ tài liệu
- **Embedding Model (HuggingFace)** — chuyển văn bản thành vector
- **LLM (OpenRouter)** — sinh câu trả lời tự nhiên từ ngữ cảnh

---

## 📋 Mục lục

- [Kiến trúc tổng quan](#kiến-trúc-tổng-quan)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Cấu hình](#cấu-hình)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
  - [1. Tạo Vector Database](#1-tạo-vector-database)
  - [2. Chạy Chatbot](#2-chạy-chatbot)
  - [3. Chạy batch câu hỏi](#3-chạy-batch-câu-hỏi)
- [Cấu trúc dữ liệu](#cấu-trúc-dữ-liệu)
- [Cấu trúc source code](#cấu-trúc-source-code)
- [API Providers](#api-providers)
- [FAQ & Troubleshooting](#faq--troubleshooting)
- [Đóng góp](#đóng-góp)

---

## 📐 Kiến trúc tổng quan

```
THU THẬP & XỬ LÝ DỮ LIỆU (1 lần)
  craw_data_from_web.ipynb  ──→  Clean_data.ipynb  ──→  File JSON (segmented)
                                                              │
                                                              ▼
                                              create_chroma_db.py ──→ ChromaDB (vector store)

CHATBOT (mỗi lần dùng)
  User Query ──→ retrieve_documents() ──→ Context ──→ create_prompt() ──→ call_llm_api() ──→ Response
                      ▲                                ▲                         ▲
                  ChromaDB                        config.yaml                  OpenRouter
              (similarity search)                (template)                (LLM API)
```

**Chi tiết từng bước:**

1. **User** nhập câu hỏi trên giao diện Gradio
2. **Retriever** tìm top-k đoạn văn bản liên quan nhất trong ChromaDB (MMR search)
3. **Prompt Constructor** ghép ngữ cảnh + câu hỏi vào template
4. **LLM** (OpenRouter) sinh câu trả lời dựa trên ngữ cảnh
5. **Response** hiển thị lên UI

> 📖 Xem thêm: [`project_architecture.md`](./project_architecture.md) — sơ đồ kiến trúc chi tiết với luồng dữ liệu từng hàm.

---

## ⚙️ Yêu cầu hệ thống

- **Python** 3.8+ (khuyến nghị 3.10+)
- **pip** (Python package manager)
- **Git** (để clone repository)
- **(Tùy chọn) CUDA** — tăng tốc embedding model nếu có GPU NVIDIA
- **(Tùy chọn) Developer Mode trên Windows** — để tránh cảnh báo symlinks khi tải HuggingFace models

---

## 📦 Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/CTNone/RAG_NLP_build_.git
cd RAG_NLP_build_
```

### 2. Tạo môi trường ảo (khuyến nghị)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 4. Thiết lập API tokens

Xem phần [Cấu hình](#cấu-hình) bên dưới.

---

## 🔧 Cấu hình

Tất cả cấu hình được quản lý qua file [`config.yaml`](./config.yaml):

```yaml
# API Credentials
api:
  provider: "openrouter"        # LLM provider hiện tại (chỉ dùng OpenRouter)
  huggingface:
    token: "hf_your_token_here"
    embedding_model: "intfloat/multilingual-e5-large-instruct"
  openrouter:
    token: "sk-or-v1-your-key"
    model_id: "openrouter/owl-alpha"   # hoặc "openai/gpt-4o-mini", ...

# Database
db_dir: "chroma_db"

# Data Paths
data_path: "data"
data_prefix: "DATA_"              # Prefix cho các thư mục dữ liệu

# Test Questions and Answers
test_questions_dir: "data/test_questions"
test_answers_dir: "data/test_answers"

# Model Parameters
max_new_tokens: 192               # Token tối đa trong câu trả lời
temperature: 0.2                  # Độ ngẫu nhiên (0.0-1.0)
top_p: 0.92                       # Nucleus sampling
repetition_penalty: 1.2           # Giảm lặp từ

# RAG Settings
retriever_k: 10                   # Số context truy xuất cho mỗi câu hỏi

# Prompt Template
template: |
  <s>[INST] <<SYS>>
  You are a direct answer bot...
  <</SYS>>
  Information: {context}
  Question: {question}
  Answer: [/INST]
```

### Đăng ký và lấy API Token

#### OpenRouter
1. Truy cập [openrouter.ai](https://openrouter.ai) → Đăng ký
2. Vào **Dashboard** → **API Keys** → Tạo key mới
3. Nạp ít nhất **$5-$10** credits để sử dụng
4. Key có dạng: `sk-or-v1-...`

#### Hugging Face (cho embedding model)
1. Truy cập [huggingface.co](https://huggingface.co/join) → Đăng ký
2. Vào [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. Tạo token mới với quyền **read**
4. Key có dạng: `hf_...`

---

## 🚀 Hướng dẫn sử dụng

### 1. Tạo Vector Database (chạy **1 lần** đầu tiên)

Script này đọc tất cả file JSON trong `data/DATA_*/segmented/`, tạo embeddings, và lưu vào ChromaDB:

```bash
python create_chroma_db.py
```

**Kết quả:** Thư mục `chroma_db/` được tạo với file `chroma.sqlite3` bên trong.

> ⚠️ Nếu đã có `chroma_db/`, script sẽ hỏi bạn có muốn xóa và tạo lại không. Chọn `y` để tạo lại.

### 2. Chạy Chatbot

```bash
python app.py
```

Sau đó mở trình duyệt và truy cập: **http://127.0.0.1:7860**

#### Các câu hỏi mẫu (có sẵn trong UI):
- "What bank, which is the 5th largest in the US, is based in Pittsburgh?"
- "How many neighborhoods does Pittsburgh have?"
- "Who is Pittsburgh named after?"
- "What famous vaccine was developed at University of Pittsburgh in 1955?"
- "DHGQHN được thành lập khi nào?"
- "VNU-USSH được thành lập khi nào"

### 3. Chạy batch câu hỏi (tự động)

Script này đọc danh sách câu hỏi từ CSV và tự động sinh câu trả lời:

```bash
python answer_questions.py
```

- **Input:** `data/test_questions/*.csv`
- **Output:** `data/test_answers/answers_*.csv`

Mỗi file output có format:
```csv
Question,Answer
"VNU là gì?","VNU là viết tắt của Đại học Quốc gia Hà Nội..."
...
```

---

## 📁 Cấu trúc dữ liệu

```
data/
├── DATA_CMU_Pit/                    # Dữ liệu về Pittsburgh & CMU
│   ├── doc_*.json                   # Raw data (chưa xử lý)
│   ├── cleaned/doc_*_cleaned.json    # Data đã làm sạch
│   └── segmented/*.json              # Data đã chunk (sẵn sàng cho ChromaDB)
├── DATA_VNU_vn/                     # Dữ liệu về VNU (tiếng Việt)
│   └── segmented/*.json
├── DATA_VNU_en/                     # Dữ liệu về VNU (tiếng Anh)
│   └── segmented/*.json
├── test_questions/                   # File CSV chứa câu hỏi test
├── test_answers/                     # File CSV chứa câu trả lời (output)
└── test_questions_mẫu.csv           # 575 câu hỏi mẫu
```

### Format file JSON trong `segmented/`

```json
{
  "url": "https://en.wikipedia.org/wiki/Vietnam_National_University,_Hanoi",
  "title": "Vietnam National University, Hanoi - Wikipedia",
  "chunks": [
    "Vietnam National University, Hanoi (VNU)... is a public research university system...",
    "VNU was established on 16 May 1906...",
    "..."
  ]
}
```

---

## 📂 Cấu trúc source code

```
RAG_NLP_build_/
│
├── app.py                     # 🚀 Entry point — Gradio Chatbot UI
├── create_chroma_db.py        # 🗄️ Script tạo ChromaDB (chạy 1 lần)
├── answer_questions.py        # 📝 Script batch xử lý câu hỏi từ CSV
├── config.yaml                # ⚙️ Cấu hình toàn bộ hệ thống
│
├── src/
│   ├── __init__.py             # Module exports
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── config_loader.py   # Đọc file config.yaml
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   └── chroma_db.py       # ChromaDB: load, create
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   └── api.py             # Gọi OpenRouter API
│   │
│   └── retriever/
│       ├── __init__.py
│       └── document_retriever.py  # Retrieval: search, load JSON, find dirs
│
├── data/                      # Dữ liệu (xem cấu trúc ở trên)
├── chroma_db/                 # Vector database (được sinh ra)
│
├── project_architecture.md    # 🏗️ Sơ đồ kiến trúc chi tiết
├── requirements.txt           # 📦 Python dependencies
├── contributions.md           # 👥 Thành viên nhóm
├── interview_knowledge.txt    # 📚 Kiến thức phỏng vấn
└── README.md                  # 📖 Bạn đang đọc đây
```

### Chi tiết từng module

| File | Chức năng chính |
|------|----------------|
| `app.py` | Khởi tạo Gradio UI, xử lý vòng đời request |
| `create_chroma_db.py` | Đọc dữ liệu → Embed → Lưu vào ChromaDB |
| `answer_questions.py` | Đọc CSV câu hỏi → Sinh câu trả lời → Ghi CSV |
| `src/utils/config_loader.py` | `load_config()` — đọc config.yaml |
| `src/database/chroma_db.py` | `load_chroma_db()` / `create_vector_db()` |
| `src/llm/api.py` | `call_llm_api()` — gọi OpenRouter API |
| `src/retriever/document_retriever.py` | `retrieve_documents()` — MMR search |

---

## 🌐 API Provider

### OpenRouter (provider duy nhất)
- **Endpoint:** `https://openrouter.ai/api/v1/chat/completions`
- **Yêu cầu:** API key + credits ($5-10+)
- **Model gợi ý:**
  - `openrouter/owl-alpha` (free)
  - `openai/gpt-4o-mini` (~$0.15/M tokens)
  - `google/gemini-2.0-flash-001` (~$0.10/M tokens)
  - `mistralai/mistral-large` (cân bằng)

---

## ❓ FAQ & Troubleshooting

### Lỗi: `403 Forbidden - error code: 1010`

**Nguyên nhân:** API key không đúng, hết credits, hoặc URL endpoint sai.

**Fix:**
1. Kiểm tra key OpenRouter: `curl -s https://openrouter.ai/api/v1/auth/key -H "Authorization: Bearer sk-or-v1-..."`
2. Nạp thêm credits tại [openrouter.ai/activity](https://openrouter.ai/activity)
3. Đảm bảo `config.yaml` đang dùng endpoint đúng: `https://openrouter.ai/api/v1/chat/completions` (có `/api/`)

### Lỗi: ChromaDB không tìm thấy context

**Nguyên nhân:** Chưa chạy `create_chroma_db.py`.

**Fix:**
```bash
python create_chroma_db.py
```

### Warning: `hf_xet not installed`

**Ảnh hưởng:** Không ảnh hưởng đến hoạt động. Có thể bỏ qua hoặc cài:
```bash
pip install huggingface_hub[hf_xet]
```

### Warning: `Symlinks not supported on Windows`

**Ảnh hưởng:** Chỉ ảnh hưởng đến hiệu năng caching. Fix bằng cách thêm vào đầu `app.py`:
```python
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
```

### Câu trả lời sai hoặc không chính xác

1. **Kiểm tra context:** Thêm `print(contexts)` trong `get_response()` để xem ChromaDB trả về gì
2. **Tăng `retriever_k`:** Trong `config.yaml`, tăng số context (mặc định 10)
3. **Kiểm tra dữ liệu:** File JSON trong `segmented/` có chứa thông tin cần thiết không?
4. **Đổi model:** Model mạnh hơn như `gpt-4o-mini` cho kết quả tốt hơn
5. **Đổi prompt template:** Format ChatML thay vì Llama format

---

## 👥 Đóng góp

| Thành viên | Đóng góp |
|:----------:|:--------:|
| Trần An Thắng | 25% |
| Chu Thân Nhất | 25% |
| Phạm Đăng Phong | 25% |
| Cao Đặng Quốc Vương | 25% |

---

## 📚 Tài liệu tham khảo

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [OpenRouter API Documentation](https://openrouter.ai/docs)
- [HuggingFace Embedding Models](https://huggingface.co/models?pipeline_tag=feature-extraction)
- [Gradio Documentation](https://www.gradio.app/docs/)
