# Kiến trúc dự án RAG_NLP_build_

Tài liệu này mô tả chi tiết kiến trúc hiện tại của dự án RAG (Retrieval-Augmented Generation) dành cho dữ liệu về CMU/Pittsburgh và VNU (Đại học Quốc gia Hà Nội).

---

## 1. Cấu trúc thư mục dự án (Project Directory Structure)

Dưới đây là sơ đồ cấu trúc của dự án sau khi phân tích mã nguồn:

```text
RAG_NLP_build_/
├── chroma_db/                  # Vector database (Chroma DB - được tạo sau khi chạy script)
│   └── chroma.sqlite3          # File SQLite3 lưu trữ vector dữ liệu
├── data/                       # Thư mục chứa dữ liệu
│   ├── DATA_CMU_Pit/           # Dữ liệu về Pittsburgh & CMU (English)
│   │   ├── doc_*.json          # File raw dữ liệu cào
│   │   └── cleaned/            # Thư mục chứa các file JSON đã làm sạch
│   ├── DATA_VNU_en/            # Dữ liệu về VNU bằng tiếng Anh (đã segmented)
│   │   └── segmented/*.json    # Chunks đã phân đoạn sẵn để nhúng (embed)
│   ├── DATA_VNU_vn/            # Dữ liệu về VNU bằng tiếng Việt (đã segmented)
│   │   └── segmented/*.json    # Chunks đã phân đoạn sẵn để nhúng (embed)
│   ├── test_questions/         # Chứa các file CSV câu hỏi kiểm thử (ví dụ: test_questions_mẫu.csv)
│   └── test_answers/           # Thư mục tự động tạo chứa câu trả lời đầu ra (.csv)
├── src/                        # Thư mục mã nguồn cốt lõi (Core Modules)
│   ├── __init__.py             # Export các hàm chính
│   ├── database/
│   │   ├── __init__.py
│   │   └── chroma_db.py        # Quản lý tải và tạo cơ sở dữ liệu vector
│   ├── llm/
│   │   ├── __init__.py
│   │   └── api.py              # Xử lý Prompt và gọi API OpenRouter / Together
│   ├── retriever/
│   │   ├── __init__.py
│   │   └── document_retriever.py # Truy xuất tài liệu (MMR) và đọc file JSON segmented
│   └── utils/
│       ├── __init__.py
│       └── config_loader.py    # Đọc file cấu hình config.yaml
├── Clean_data.ipynb            # Notebook xử lý và làm sạch dữ liệu raw
├── craw_data_from_web.ipynb    # Notebook cào dữ liệu từ Wikipedia và các nguồn khác
├── create_chroma_db.py         # Script tạo/lập chỉ mục Vector DB từ các folder dữ liệu segmented
├── answer_questions.py         # Script kiểm thử/trả lời câu hỏi hàng loạt từ file CSV
├── app.py                      # Ứng dụng web chatbot tương tác (sử dụng Gradio)
├── config.yaml                 # File cấu hình tập trung (API key, model parameters, RAG settings)
├── requirements.txt            # Danh sách thư viện Python phụ thuộc
└── README.md                   # Hướng dẫn sử dụng dự án
```

---

## 2. Sơ đồ Pipeline Tổng quan (System Pipeline)

Kiến trúc của dự án được chia làm 3 giai đoạn độc lập nhưng liên kết chặt chẽ với nhau:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       GIAI ĐOẠN 1: TẠO VÀ LƯU VECTOR DB                    │
│                      (Chạy script: create_chroma_db.py)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Dữ liệu JSON trong data/DATA_*/segmented/                                  │
│         │                                                                   │
│         ▼                                                                   │
│  load_json_documents() ─────────────────> Trả về danh sách LangChain Document │
│  (document_retriever.py)                (Kèm metadata: title, url, chunk_id)│
│                                                     │                       │
│                                                     ▼                       │
│  create_vector_db() <───────────────────────────────┘                       │
│  (chroma_db.py)                                                             │
│         │                                                                   │
│         ├─ Embedding model: HuggingFaceEmbeddings                           │
│         │  (intfloat/multilingual-e5-large-instruct)                       │
│         │                                                                   │
│         ▼                                                                   │
│  Lưu trữ vật lý xuống thư mục chroma_db/ (sqlite)                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    GIAI ĐOẠN 2: CHATBOT TƯƠNG TÁC TRỰC TUYẾN                │
│                        (Chạy ứng dụng: app.py)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [Giao diện Gradio ChatInterface]                                           │
│         │                                                                   │
│  User gửi: Question                                                         │
│         │                                                                   │
│         ▼                                                                   │
│  retrieve_documents() ──> Truy vấn Chroma DB (Thuật toán MMR)                │
│                           Lấy k = 10 tài liệu từ fetch_k = 20 ứng viên      │
│                                 │                                           │
│                                 ▼                                           │
│  create_prompt() <──────────────┘                                           │
│  Format prompt theo Template cấu hình (Llama / ChatML Format)               │
│         │                                                                   │
│         ▼                                                                   │
│  call_llm_api() ────────> Chọn API Provider (OpenRouter / Together)          │
│                           Gửi kèm prompt và lịch sử hội thoại (history)     │
│                                 │                                           │
│                                 ▼                                           │
│  Trả về câu trả lời cho User trên Gradio UI                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    GIAI ĐOẠN 3: KIỂM THỬ HÀNG LOẠT QUA FILE CSV             │
│                    (Chạy script: answer_questions.py)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Đọc danh sách câu hỏi từ CSV trong data/test_questions/                    │
│         │                                                                   │
│         ▼                                                                   │
│  Duyệt qua từng câu hỏi:                                                    │
│    - Gọi retrieve_documents() từ DB                                         │
│    - Tạo prompt tương ứng qua create_prompt()                               │
│    - Gọi LLM qua call_llm_api() để lấy câu trả lời                          │
│         │                                                                   │
│         ▼                                                                   │
│  Lưu toàn bộ cặp Question-Answer vào file CSV mới trong data/test_answers/   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Chi tiết các file & API chính

### 📄 `app.py` — Giao diện Chatbot Gradio
* **`initialize_app()`** (Dòng 38-63): Đọc file `config.yaml`, xác định provider, in thông tin cấu hình và tải cơ sở dữ liệu Chroma DB vào bộ nhớ toàn cục.
* **`get_response(message, history)`** (Dòng 18-35): Nhận tin nhắn từ giao diện, thực hiện truy xuất tài liệu liên quan, dựng prompt và gọi API để lấy câu trả lời.
* **`demo = gr.ChatInterface(...)`** (Dòng 77-84): Thiết lập giao diện trò chuyện Gradio với các câu hỏi mẫu gợi ý (ví dụ về Pittsburgh, CMU và VNU).

### 📄 `create_chroma_db.py` — Khởi tạo Vector Database
* **`main()`** (Dòng 17-124):
  1. Tải cấu hình và cấu hình thiết bị phần cứng (Ưu tiên CUDA GPU nếu có, ngược lại dùng CPU).
  2. Tự động quét và tìm các thư mục chứa tiền tố `DATA_` trong thư mục `data`.
  3. Quét đệ quy tất cả các file `.json` trong thư mục con `segmented/`.
  4. Đọc dữ liệu thô và chuyển thành danh sách các đối tượng `Document`.
  5. Kiểm tra sự tồn tại của database: Nếu có sẵn, script sẽ hỏi người dùng có muốn ghi đè (`y/n`) hay không. Nếu có hoặc chưa tồn tại, nó sẽ thực hiện khởi tạo và ghi đè mới.

### 📄 `answer_questions.py` — Đánh giá & Trả lời câu hỏi hàng loạt
* **`load_questions_from_csv(file_path)`** (Dòng 23-36): Đọc cột đầu tiên của file CSV chứa câu hỏi để xử lý.
* **`get_answer(question, db, config)`** (Dòng 39-55): Đóng gói luồng RAG cơ bản (Truy xuất -> Tạo prompt -> Gọi API).
* **`process_question_file(file_path, db, config)`** (Dòng 57-72): Lặp qua tập câu hỏi của file, gọi `get_answer` và dừng 0.5 giây giữa các lượt gọi (`time.sleep(0.5)`) để tránh bị rate limit từ API.
* **`save_answers_to_csv(...)`** (Dòng 75-90): Ghi danh sách câu hỏi và câu trả lời sinh ra vào file định dạng CSV mới.
* **`main()`** (Dòng 92-131): Đọc toàn bộ các file `.csv` từ thư mục `test_questions_dir` và lưu kết quả vào thư mục `test_answers_dir`.

### 📄 `config.yaml` — Cấu hình tập trung
* **`api.provider`**: Chọn provider API LLM (`"openrouter"` hoặc `"together"`).
* **`api.huggingface`**: Cấu hình token HuggingFace và mô hình embedding (`intfloat/multilingual-e5-large-instruct`).
* **`api.openrouter`**: Chứa token và model ID (mặc định: `openrouter/owl-alpha`).
* **`db_dir`**, **`data_path`**, **`data_prefix`**: Cấu hình đường dẫn lưu trữ DB và thư mục chứa dữ liệu thô.
* **`test_questions_dir`**, **`test_answers_dir`**: Đường dẫn cho script chạy thử nghiệm.
* **Các tham số sinh văn bản**: `max_new_tokens` (192), `temperature` (0.2), `top_p` (0.92), `repetition_penalty` (1.2).
* **Prompt Template**: Cấu hình sẵn prompt dạng Llama instruction.

### 📄 `src/database/chroma_db.py` — Quản lý Vector DB
* **`load_chroma_db(config)`** (Dòng 10-24): Tải mô hình embedding bằng thiết bị phù hợp, kết nối và trả về đối tượng `Chroma`.
* **`create_vector_db(documents, db_path, embeddings_model, device)`** (Dòng 27-41): Sử dụng phương thức `Chroma.from_documents` để lưu trữ embeddings và văn bản trực tiếp xuống thư mục chỉ định trên ổ đĩa cứng.

### 📄 `src/retriever/document_retriever.py` — Tìm kiếm tài liệu đa dạng
* **`retrieve_documents(db, query, k)`** (Dòng 9-17): Sử dụng thuật toán truy xuất **MMR (Maximal Marginal Relevance)**.
  * *Chi tiết:* Tìm kiếm `fetch_k = 20` đoạn văn bản có độ tương đồng cao nhất, sau đó chọn lọc ra `k = 10` đoạn có nội dung đa dạng nhất, tránh trùng lặp thông tin nhằm tối ưu hóa ngữ cảnh đưa vào LLM.
* **`load_json_documents(file_path)`** (Dòng 25-54): Đọc dữ liệu từ file JSON phân đoạn, trích xuất danh sách `chunks` và gán metadata (`source`, `url`, `title`, `chunk_id`).
* **`get_data_directories(base_path, prefix)`** (Dòng 57-66): Lọc ra các thư mục con bắt đầu bằng tiền tố thích hợp.

### 📄 `src/llm/api.py` — Xử lý giao tiếp với LLM API
* **`create_prompt(...)`** (Dòng 16-20): Ghép nối các ngữ cảnh được tìm thấy thành chuỗi văn bản và định dạng vào Prompt Template.
* **`_prepare_messages(...)`** (Dòng 23-49): Chuyển đổi lịch sử chat (từ Gradio, hỗ trợ nhiều kiểu định dạng như tuple, list, dict) thành danh sách tin nhắn chuẩn OpenAI API `[{"role": "user/assistant", "content": "..."}]`.
* **`_call_openrouter(...)`** (Dòng 52-85): Thực hiện gửi yêu cầu POST qua thư viện `urllib.request` đến API OpenRouter để lấy dữ liệu văn bản.
* **`_call_together(...)`** (Dòng 88-104): Khởi tạo Client Together và gọi phương thức chat completions bằng thư viện SDK chính thức (`together`).
* **`call_llm_api(...)`** (Dòng 107-113): Bộ định tuyến lựa chọn hàm xử lý tương ứng dựa trên cấu hình provider.

### 📄 `src/utils/config_loader.py` — Trình tải file YAML
* **`load_config(config_path)`** (Dòng 7-25): Đọc file YAML và nạp các biến cấu hình.

---

## 4. Chi tiết luồng dữ liệu (Data Flow details)

### A. Quá trình Indexing Dữ liệu
```text
[File JSON Phân đoạn]
       │
       ▼ (Đọc cấu trúc file JSON)
[Trích xuất chunks, url, title]
       │
       ▼ (Bọc bằng LangChain Document)
[Document(page_content=chunk, metadata={source, url, title, chunk_id})]
       │
       ▼ (HuggingFaceEmbeddings: intfloat/multilingual-e5-large-instruct)
[Vector embeddings biểu diễn ngữ nghĩa (768-dim)]
       │
       ▼ (Chroma.from_documents)
[Lưu trữ xuống chroma_db/chroma.sqlite3]
```

### B. Quá trình sinh câu trả lời RAG
```text
               User Question (ví dụ: "ĐHQGHN thành lập khi nào?")
                            │
                            ▼
            Chroma.as_retriever(search_type="mmr")
                            │
      ┌─────────────────────┴─────────────────────┐
      ▼                                           ▼
Lọc ra 20 ứng viên tương đồng nhất            Chọn ra 10 đoạn đa dạng nhất
(Similarity Search)                           (Maximal Marginal Relevance)
      │                                           │
      └─────────────────────┬─────────────────────┘
                            │
                            ▼
                    List[Context text]
                            │
                            ▼
                     create_prompt()
            (Format vào Prompt Template trong config)
                            │
                            ▼
                      call_llm_api()
            (Gọi OpenRouter API)
                            │
                            ▼
                  [Câu trả lời trả về]
```

---

## 5. Các phát hiện & Vấn đề kỹ thuật cần lưu ý (Bug & Optimization Notes)

### 1. ✅ Đã sửa: Lỗi nạp biến môi trường cho Token HuggingFace trong `config_loader.py`
* **Vấn đề trước đây:** Hàm `load_config` kiểm tra khóa `"huggingface_api_token"` ở cấp ngoài cùng của config, trong khi cấu trúc của YAML là lồng nhau (`config["api"]["huggingface"]["token"]`).
* **Giải pháp đã thực hiện:** Sửa lại logic kiểm tra và gán vào đúng vị trí cấu hình lồng nhau. Bây giờ biến môi trường `HUGGINGFACEHUB_API_TOKEN` sẽ ghi đè chính xác token trong config nếu được khai báo.

### 2. ✅ Đã xử lý: Lỗi KeyError: 'together' (Đã loại bỏ Together AI)
* **Vấn đề trước đây:** Hàm `call_llm_api` hỗ trợ provider `"together"` nhưng `config.yaml` không khai báo thông số kết nối, dẫn đến lỗi `KeyError: 'together'`.
* **Giải pháp đã thực hiện:** Do dự án không còn sử dụng Together AI, toàn bộ logic import, hàm `_call_together` và tùy chọn provider `"together"` đã được loại bỏ hoàn toàn khỏi mã nguồn (`api.py` và `app.py`). Hệ thống mặc định sử dụng OpenRouter.

### 3. 💡 Khuyến nghị về Prompt Template & Model
* **Vấn đề:** Prompt template hiện tại đang sử dụng cấu trúc của Llama `<s>[INST] <<SYS>> ... <</SYS>> ... [/INST]`. Tuy nhiên, model đang được chọn làm mặc định trên OpenRouter lại là `openrouter/owl-alpha`.
* **Khuyến nghị:** Model `owl-alpha` có thể không tuân thủ tốt cấu trúc chỉ dẫn này. Nên chuyển sang sử dụng định dạng ChatML phổ biến hoặc cấu hình các dòng model có chất lượng tốt hơn như `gemini-2.0-flash`, `gpt-4o-mini` để nâng cao chất lượng câu trả lời.
