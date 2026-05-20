# Kiến trúc dự án RAG_NLP_build_

## 1. Sơ đồ tổng quan (Pipeline)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GIAI ĐOẠN 1: TẠO DỮ LIỆU                           │
│                     (create_chroma_db.py - chạy 1 lần)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  data/                                                                      │
│  ├── DATA_CMU_Pit/ (dữ liệu về Pittsburgh & CMU)                          │
│  │   ├── doc_*.json (raw)                                                  │
│  │   └── cleaned/doc_*_cleaned.json (đã làm sạch)                         │
│  ├── DATA_VNU_vn/ (dữ liệu về VNU - tiếng Việt)                          │
│  │   └── segmented/*.json (đã chunk)                                      │
│  └── DATA_VNU_en/ (dữ liệu về VNU - tiếng Anh)                           │
│       └── segmented/*.json (đã chunk)                                     │
│                                                                             │
│  ┌──────────────────────────────┐                                          │
│  │ load_json_documents()       │ ← Đọc từng file JSON, parse chunks       │
│  │ (document_retriever.py:18)  │   → List[Document] (langchain)            │
│  └──────────────┬──────────────┘                                          │
│                 │                                                          │
│                 ▼                                                          │
│  ┌──────────────────────────────┐                                          │
│  │ HuggingFaceEmbeddings()      │ ← intfloat/multilingual-e5-large-instruct│
│  │ (chroma_db.py:17)           │                                            │
│  └──────────────┬──────────────┘                                            │
│                 │                                                          │
│                 ▼                                                          │
│  ┌──────────────────────────────┐                                          │
│  │ Chroma.from_documents()     │ → chroma_db/ (vector store)              │
│  │ (chroma_db.py:37)          │                                            │
│  └──────────────────────────────┘                                          │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                     GIAI ĐOẠN 2: CHẠY CHATBOT                              │
│                     (python app.py - mỗi lần dùng)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────┐                                          │
│  │ load_config("config.yaml")  │ ← Đọc: provider, model, token, template │
│  │ (config_loader.py:7)        │                                            │
│  └──────────────┬──────────────┘                                            │
│                 │                                                          │
│                 ▼                                                          │
│  ┌──────────────────────────────┐                                          │
│  │ load_chroma_db(config)      │ ← Load Chroma từ chroma_db/              │
│  │ (chroma_db.py:10)           │   + HuggingFaceEmbeddings                │
│  └──────────────┬──────────────┘                                            │
│                 │                                                          │
│                 ▼                                                          │
│  ┌────────── CHAT LOOP (Gradio) ──────────────────────────────────┐       │
│  │                                                                │       │
│  │  User: "DHGQHN được thành lập khi nào?"                        │       │
│  │         │                                                      │       │
│  │         ▼                                                      │       │
│  │  ┌──────────────────────┐                                      │       │
│  │  │ retrieve_documents()│ ← Similarity/MMR search trong Chroma │       │
│  │  │ (document_retriever:9)│   fetch_k=20, chọn k=10 docs      │       │
│  │  └──────────┬───────────┘                                      │       │
│  │             │ → List[str] contexts (vd: 10 đoạn text)           │       │
│  │             ▼                                                  │       │
│  │  ┌──────────────────────┐                                      │       │
│  │  │ create_prompt()     │ ← PromptTemplate + {context,question}│       │
│  │  │ (api.py:16)          │   → str prompt (formatted)           │       │
│  │  └──────────┬───────────┘                                      │       │
│  │             │                                                  │       │
│  │             ▼ (gửi đến LLM)                                   │       │
│  │  ┌──────────────────────┐                                      │       │
│  │  │ call_llm_api()      │                                      │       │
│  │  │ (api.py:105)         │                                      │       │
│  │  │  provider?           │                                      │       │
│  │  │  ├─ openrouter ──→ _call_openrouter() → OpenAI API         │       │
│  │  │  └─ together ──→ _call_together() → Together.ai API       │       │
│  │  └──────────┬───────────┘                                      │       │
│  │             │ → str response                                  │       │
│  │             ▼                                                  │       │
│  │  Trả về Gradio UI → User thấy câu trả lời                    │       │
│  │                                                                │       │
│  └────────────────────────────────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Chi tiết các file & hàm

### 📄 `app.py` — Entry point (Giao diện Gradio)

| Hàm | Dòng | Chức năng |
|-----|------|-----------|
| `initialize_app()` | 38-63 | Khởi tạo: load config, load Chroma DB |
| `get_response(message, history)` | 18-35 | Xử lý 1 câu hỏi: retrieve → prompt → LLM |
| `demo = gr.ChatInterface(...)` | 77-84 | Tạo giao diện Gradio với examples |

**Luồng dữ liệu trong 1 request:**
```
message → retrieve_documents() → contexts (List[str])
                                    ↓
                              create_prompt() → prompt (str)
                                    ↓
                              call_llm_api() → response (str)
                                    ↓
                              return response → Gradio UI
```

### 📄 `config.yaml` — Cấu hình

| Key | Giá trị | Chức năng |
|-----|---------|-----------|
| `api.provider` | `"openrouter"` | Provider LLM hiện tại |
| `api.openrouter.token` | `"sk-or-v1-..."` | API key OpenRouter |
| `api.openrouter.model_id` | `"openrouter/owl-alpha"` | Model đang dùng |
| `api.huggingface.embedding_model` | `"intfloat/multilingual-e5-large-instruct"` | Embedding model |
| `retriever_k` | `10` | Số context trả về |
| `template` | Llama format | Prompt template |

### 📄 `src/__init__.py` — Module exports

Export tất cả các hàm chính:
- `load_config`, `load_chroma_db`, `create_vector_db`
- `retrieve_documents`, `load_json_documents`, `get_data_directories`
- `call_llm_api`, `create_prompt`

### 📄 `src/utils/config_loader.py` — Đọc config

| Hàm | Dòng | Chức năng |
|-----|------|-----------|
| `load_config(config_path)` | 7-25 | Đọc `config.yaml` → dict |

### 📄 `src/database/chroma_db.py` — Vector Database

| Hàm | Dòng | Chức năng |
|-----|------|-----------|
| `load_chroma_db(config)` | 10-24 | Load Chroma từ `chroma_db/` + embedding |
| `create_vector_db(documents, db_path, ...)` | 27-39 | Tạo Chroma từ documents |

**Đầu vào/ra:**
- `load_chroma_db`: config → Chroma object
- `create_vector_db`: List[Document] + path → lưu xuống disk

### 📄 `src/retriever/document_retriever.py` — Truy vấn

| Hàm | Dòng | Chức năng |
|-----|------|-----------|
| `retrieve_documents(db, query, k)` | 9-17 | Search Chroma → List[str] context |
| `load_json_documents(file_path)` | 25-47 | Parse file JSON → List[Document] |
| `get_data_directories(base_path, prefix)` | 57-66 | Tìm thư mục DATA_* |

**Đầu vào/ra:**
- `retrieve_documents`: Chroma DB + câu hỏi → 10 đoạn text

### 📄 `src/llm/api.py` — Gọi API LLM

| Hàm | Dòng | Chức năng |
|-----|------|-----------|
| `create_prompt(question, contexts, template)` | 16-20 | Format prompt từ template |
| `_prepare_messages(history, prompt)` | 23-49 | Convert history → OpenAI message format |
| `_call_openrouter(prompt, config, history)` | 52-83 | Gọi OpenRouter API (REST) |
| `_call_together(prompt, config, history)` | 86-102 | Gọi Together API (SDK) |
| `call_llm_api(prompt, config, history)` | 105-111 | Router: chọn provider |

**Đầu vào/ra:**
- `call_llm_api`: prompt string → response string

### 📄 `create_chroma_db.py` — Script tạo DB (chạy 1 lần)

**Luồng:**
```
1. load_config() → config dict
2. get_data_directories() → [data/DATA_CMU_Pit, data/DATA_VNU_vn, data/DATA_VNU_en]
3. Với mỗi thư mục:
   a. Tìm all file .json trong **/segmented/**
   b. Với mỗi file: load_json_documents() → Document[]
4. Gộp tất cả Document → create_vector_db()
5. → chroma_db/chroma.sqlite3 (vector store)
```

---

## 3. Luồng dữ liệu chi tiết

### A. Từ file JSON → Chroma DB

```
File JSON (raw/cleaned/segmented)
  │
  ├── "url": "https://..."
  ├── "title": "..."
  ├── "content": "..." (full text)
  └── "chunks": ["chunk1", "chunk2", ...] (đã chia nhỏ)
       │
       ▼
load_json_documents()
  → Document(page_content=chunk, metadata={source, url, title, chunk_id})
       │
       ▼
HuggingFaceEmbeddings("intfloat/multilingual-e5-large-instruct")
  → Vector embeddings (768-dim)
       │
       ▼
Chroma.from_documents()
  → chroma_db/chroma.sqlite3 (vector database)
```

### B. Từ câu hỏi → Câu trả lời

```
User: "DHGQHN được thành lập khi nào?"
       │
       ▼
retrieve_documents(db, query, k=10)
  → search_type="mmr" (MMR: diversity search)
  → fetch_k=20 docs → chọn 10 docs đa dạng nhất
       │
       ▼
[Context 1] "...Founded in July, 2002, VNU-IS..."
[Context 2] "...VNU was established on 16 May 1906..."
...
[Context 10] "...Đại học Quốc gia Hà Nội (ĐHQGHN)..."
       │
       ▼
create_prompt(question, contexts, template)
  → Prompt:
    <s>[INST] <<SYS>>
    You are a direct answer bot...
    Rules:
    - Answer in SAME LANGUAGE
    - Never explain reasoning
    - ...
    <</SYS>>
    Information: [Context 1]...[Context 10]
    Question: DHGQHN được thành lập khi nào?
    Answer: [/INST]
       │
       ▼
call_llm_api(prompt, config, history)
  → _call_openrouter()
  → POST https://openrouter.ai/api/v1/chat/completions
  → model: openrouter/owl-alpha
       │
       ▼
Response: "Đại học Quốc gia Hà Nội được thành lập vào ngày 16 tháng 5 năm 1906"
```

---

## 4. Các vấn đề hiện tại & hướng tối ưu

### ❌ Vấn đề 1: Chroma DB chưa được tạo
- **Nguyên nhân:** File `chroma_db/` không tồn tại trong project → chưa chạy `create_chroma_db.py`
- **Hậu quả:** `retrieve_documents()` trả về 0 context → model trả lời mù quáng
- **Fix:** `python create_chroma_db.py`

### ❌ Vấn đề 2: Prompt format không phù hợp
- **Loại:** Llama format `<s>[INST]...[/INST]>`
- **Model dùng:** OWL Alpha (có thể không hỗ trợ tốt)
- **Nên đổi sang:** ChatML format

### ❌ Vấn đề 3: Retriever dùng similarity (đã fix MMR)
- Trước: `similarity` → trùng context
- Sau: `mmr` + `fetch_k=20` → context đa dạng hơn

### ❌ Vấn đề 4: Model OWL Alpha yếu
- Model nhẹ, instruction-following kém
- Nên thử: `gpt-4o-mini`, `gemini-2.0-flash`

---

## 5. File dependencies map

```
app.py
  ├── src/__init__.py
  │   ├── src/utils/config_loader.py  →  load_config()
  │   ├── src/database/chroma_db.py   →  load_chroma_db()
  │   ├── src/llm/api.py             →  call_llm_api(), create_prompt()
  │   └── src/retriever/__init__.py
  │       └── src/retriever/document_retriever.py  →  retrieve_documents()
  └── config.yaml

create_chroma_db.py
  ├── src/__init__.py
  │   ├── src/utils/config_loader.py  →  load_config()
  │   ├── src/database/chroma_db.py   →  create_vector_db()
  │   └── src/retriever/__init__.py
  │       └── src/retriever/document_retriever.py  →  load_json_documents(), get_data_directories()
  └── config.yaml
  └── data/*/segmented/*.json
