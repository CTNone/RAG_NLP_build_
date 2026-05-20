# 🎓 Hướng Dẫn Phỏng Vấn Dự Án RAG_NLP_build_ - 30 Câu Hỏi Toàn Diện

> **Chuẩn bị cho phỏng vấn dự án RAG Chatbot**
> 
> Tài liệu này chứa 30 câu hỏi được sắp xếp tăng dần theo độ khó, từ cơ bản đến nâng cao.

---

## 📊 Thống Kê Dự Án

- **Tên dự án:** RAG_NLP_build_
- **Mô tả:** Build hệ thống truy xuất dữ liệu (Retrieval-Augmented Generation)
- **Ngôn ngữ:** 90.3% Jupyter Notebook, 9.6% Python, 0.1% Dockerfile
- **Chủ đề:** NLP, RAG, Chatbot, Vector Database, LLM

---

## 🟢 PHẦN 1: CÂU HỎI CƠ BẢN (Cấp độ 1-10)

### 1️⃣ **Câu 1: Dự án này là gì?**

**Trả lời:**
- Đây là một hệ thống **RAG (Retrieval-Augmented Generation)** - kết hợp truy xuất thông tin và sinh tế bản.
- Giúp trả lời câu hỏi về **Pittsburgh, Carnegie Mellon University (CMU), và Đại học Quốc gia Hà Nội (VNU)**.
- Sử dụng **ChromaDB** để lưu trữ vector embeddings, **HuggingFace** để tạo embeddings, và **OpenRouter API** để gọi LLM.
- Có giao diện web **Gradio** để tương tác.

---

### 2️⃣ **Câu 2: RAG là gì? Tại sao cần RAG?**

**Trả lời:**
- **RAG = Retrieval-Augmented Generation**
- **Retrieval:** Tìm kiếm tài liệu liên quan từ knowledge base
- **Augmented:** Tăng cường prompt bằng context từ tài liệu
- **Generation:** Sinh câu trả lời bằng LLM dựa trên context

**Tại sao cần:**
- LLM có kiến thức cắt ngắn, chỉ có thể trả lời từ training data
- RAG cung cấp thông tin cập nhật từ vector database
- Giảm hallucination (LLM sinh ra thông tin sai)
- Tiết kiệm chi phí (không cần fine-tune model)

---

### 3️⃣ **Câu 3: Các thành phần chính của dự án là gì?**

**Trả lời:**

| Thành phần | Chức năng |
|-----------|---------|
| **ChromaDB** | Vector database - lưu trữ embeddings, hỗ trợ similarity search |
| **HuggingFace Embeddings** | Mô hình embedding (multilingual-e5-large-instruct) |
| **OpenRouter API** | LLM provider - sinh câu trả lời |
| **Gradio** | Web UI - giao diện chatbot |
| **Python Scripts** | Xử lý dữ liệu, tạo DB, batch processing |

---

### 4️⃣ **Câu 4: Luồng hoạt động chính của chatbot là gì?**

**Trả lời:**

```
User Query
    ↓
[Embedding] → Vector query
    ↓
[Retrieval] → Tìm top-k documents từ ChromaDB (MMR search)
    ↓
[Context] → Ghép context vào prompt template
    ↓
[LLM Call] → Gọi OpenRouter API
    ↓
[Response] → Trả lời cho user
    ↓
[Display] → Hiển thị trên Gradio UI
```

---

### 5️⃣ **Câu 5: File cấu hình config.yaml chứa những gì?**

**Trả lời:**
- **API Credentials:** OpenRouter token, HuggingFace token
- **Model Selection:** Model ID (openrouter/owl-alpha, embedding model)
- **Database Path:** Đường dẫn ChromaDB
- **RAG Settings:** retriever_k = 10 (số document lấy ra)
- **Model Parameters:** temperature, top_p, max_new_tokens, repetition_penalty
- **Prompt Template:** Định dạng prompt (Llama instruction format)
- **Data Paths:** Đường dẫn thư mục dữ liệu, test questions, test answers

---

### 6️⃣ **Câu 6: Cấu trúc thư mục dữ liệu như thế nào?**

**Trả lời:**

```
data/
├── DATA_CMU_Pit/           # Dữ liệu Pittsburgh & CMU
│   ├── doc_*.json          # Raw data
│   ├── cleaned/            # Dữ liệu đã làm sạch
│   └── segmented/          # Dữ liệu đã chunk (sẵn sàng embedding)
├── DATA_VNU_vn/            # Dữ liệu VNU tiếng Việt
│   └── segmented/
├── DATA_VNU_en/            # Dữ liệu VNU tiếng Anh
│   └── segmented/
├── test_questions/         # CSV chứa câu hỏi test
└── test_answers/           # CSV kết quả (tự động tạo)
```

---

### 7️⃣ **Câu 7: Các file Python chính là gì?**

**Trả lời:**
- **app.py:** Gradio UI - entry point chính
- **create_chroma_db.py:** Script tạo vector database (chạy 1 lần)
- **answer_questions.py:** Batch processing - trả lời nhiều câu hỏi từ CSV
- **config_loader.py:** Đọc config.yaml
- **chroma_db.py:** Quản lý ChromaDB (load, create)
- **document_retriever.py:** Truy xuất tài liệu (MMR search)
- **api.py:** Gọi OpenRouter API

---

### 8️⃣ **Câu 8: MMR là gì và tại sao sử dụng MMR?**

**Trả lời:**
- **MMR = Maximal Marginal Relevance**
- **Cách hoạt động:**
  1. Tìm fetch_k = 20 documents có similarity cao nhất
  2. Từ 20 documents này, chọn ra k = 10 documents đa dạng nhất
- **Tại sao:** Tránh trùng lặp, đảm bảo tính đa dạng của context

---

### 9️⃣ **Câu 9: Embedding model là gì và dự án sử dụng model nào?**

**Trả lời:**
- **Embedding Model:** Chuyển text thành vector (768-dim) biểu diễn ngữ nghĩa
- **Dự án sử dụng:** `intfloat/multilingual-e5-large-instruct`
- **Lý do:**
  - Hỗ trợ multilingual (Anh, Việt)
  - Chất lượng cao, instruction-following tốt
  - Đã training trên 50+ ngôn ngữ

---

### 🔟 **Câu 10: ChromaDB là gì? Tại sao sử dụng ChromaDB?**

**Trả lời:**
- **ChromaDB:** Vector database - lưu trữ embeddings và metadata
- **Tại sao:**
  - Dễ sử dụng, tích hợp tốt với LangChain
  - Hỗ trợ MMR search (Maximal Marginal Relevance)
  - Lưu trữ persistent (SQLite - không cần server riêng)
  - Tốc độ query nhanh
  - Hỗ trợ metadata filtering

---

## 🟡 PHẦN 2: CÂU HỎI TRUNG BÌNH (Cấp độ 11-20)

### 1️⃣1️⃣ **Câu 11: Quá trình tạo vector database (create_chroma_db.py) diễn ra như thế nào?**

**Trả lời:**
1. **Load config:** Đọc config.yaml
2. **Detect GPU/CPU:** Ưu tiên CUDA nếu có GPU NVIDIA
3. **Auto-discover directories:** Tìm tất cả folder `DATA_*` trong thư mục `data/`
4. **Scan JSON files:** Quét recursive tất cả `.json` trong `segmented/`
5. **Load documents:** Chuyển từ JSON thành LangChain Document objects
6. **Create embeddings:** Sử dụng HuggingFaceEmbeddings tạo vectors
7. **Save to DB:** Lưu vào ChromaDB dùng `Chroma.from_documents()`
8. **Persist:** Lưu xuống `chroma_db/chroma.sqlite3`

---

### 1️⃣2️⃣ **Câu 12: Format file JSON segmented như thế nào?**

**Trả lời:**

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

**Cấu trúc:**
- `url`: Link nguồn
- `title`: Tiêu đề tài liệu
- `chunks`: Danh sách các đoạn văn bản (text segments)

---

### 1️⃣3️⃣ **Câu 13: Hàm retrieve_documents() trong document_retriever.py hoạt động thế nào?**

**Trả lời:**

```python
def retrieve_documents(db, query, k=3, config=None):
    # 1. Expand query (thêm alias cho Vietnamese acronyms)
    expanded_query = _expand_query(query)
    
    # 2. Khởi tạo retriever với MMR search
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 20}
    )
    
    # 3. Invoke retriever
    docs = retriever.invoke(expanded_query)
    
    # 4. Format kết quả thành RetrievedContext objects
    return [_build_context(doc, i+1) for i, doc in enumerate(docs)]
```

---

### 1️⃣4️⃣ **Câu 14: Hàm create_prompt() làm gì?**

**Trả lời:**
- **Input:** question, contexts (list), template
- **Process:**
  1. Apply context budget (giới hạn tổng độ dài)
  2. Format từng context với citation ID
  3. Ghép tất cả contexts thành một text
  4. Substitute vào template: `{context}` và `{question}`
- **Output:** Prompt string sẵn sàng gửi đến LLM
- **Mục đích:** Cấu trúc hóa input cho LLM, thêm citations

---

### 1️⃣5️⃣ **Câu 15: Hàm call_llm_api() hoạt động thế nào?**

**Trả lời:**

```python
def call_llm_api(prompt, config, history=None, stream=False):
    provider = config["api"]["provider"]  # "openrouter"
    
    if provider == "openrouter":
        return _call_openrouter(prompt, config, history)
    else:
        raise ValueError(f"Unknown provider: {provider}")
```

**_call_openrouter():**
1. Extract API key, model ID từ config
2. Prepare headers (Authorization, Content-Type)
3. Prepare messages từ history + prompt
4. Create JSON body với model, messages, temperature, top_p
5. POST request tới `https://openrouter.ai/api/v1/chat/completions`
6. Parse response JSON
7. Return message content

---

### 1️⃣6️⃣ **Câu 16: Gradio ChatInterface được setup như thế nào trong app.py?**

**Trả lời:**

```python
demo = gr.ChatInterface(
    fn=get_response,  # Hàm xử lý input
    title="RAG Chatbot về Pittsburgh / CMU và VNU",
    description="Chatbot có khả năng truy vấn dữ liệu...",
    examples=[  # Các câu hỏi mẫu
        "What bank is based in Pittsburgh?",
        "DHQGHN được thành lập khi nào?",
        ...
    ],
)
```

**Tính năng:**
- `fn=get_response`: Function xử lý chat
- `examples`: Gợi ý câu hỏi
- Automatic message history tracking
- Real-time response display

---

### 1️⃣7️⃣ **Câu 17: Hàm get_response() trong app.py có role gì?**

**Trả lời:**
- **Input:** User message + chat history
- **Process:**
  1. Validate: Check app_config, app_db đã load
  2. **Retrieve:** Gọi retrieve_documents()
  3. **Create prompt:** Gọi create_prompt()
  4. **Call LLM:** Gọi call_llm_api()
  5. **Timing:** Đo thời gian retrieval và LLM call
  6. **Debug:** In thông tin context nếu debug enabled
- **Output:** Câu trả lời string
- **Error handling:** Catch exceptions, return error message

---

### 1️⃣8️⃣ **Câu 18: Batch processing (answer_questions.py) làm gì?**

**Trả lời:**
- **Input:** CSV files trong `data/test_questions/`
- **Process:**
  1. Load config và ChromaDB
  2. Đọc từng file CSV
  3. Duyệt qua mỗi câu hỏi
  4. Gọi RAG pipeline (retrieve → create_prompt → call_llm)
  5. Sleep 0.5s giữa các lần (rate limiting)
  6. Collect answers
- **Output:** CSV mới trong `data/test_answers/` với columns: Question, Answer
- **Mục đích:** Kiểm thử hàng loạt, evaluation

---

### 1️⃣9️⃣ **Câu 19: Initialize_app() trong app.py có trách nhiệm gì?**

**Trả lời:**
- **Mục đích:** Khởi tạo ứng dụng khi startup
- **Steps:**
  1. Load config.yaml bằng load_config()
  2. Validate config (check provider, model_id, embedding_model)
  3. Load ChromaDB vào bộ nhớ bằng load_chroma_db()
  4. Gán vào global variables: app_config, app_db
  5. Print thông tin startup
- **Return:** True nếu thành công, False nếu lỗi
- **Timing:** Chạy duy nhất 1 lần trước khi demo.launch()

---

### 2️⃣0️⃣ **Câu 20: Quá trình nhúng (embedding) dữ liệu diễn ra thế nào?**

**Trả lời:**
1. **Load model:** `HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")`
2. **Text to vector:** Mỗi chunk text được chuyển thành vector 768-dim
3. **Store in DB:** Vector + metadata (url, title, chunk_id) được lưu ChromaDB
4. **Persistence:** SQLite lưu toàn bộ vào `chroma_db/chroma.sqlite3`
5. **Query time:** User query cũng được embedding theo model tương tự
6. **Similarity:** So sánh vector similarity (cosine) để tìm documents liên quan

---

## 🔴 PHẦN 3: CÂU HỎI NÂN CAO (Cấp độ 21-30)

### 2️⃣1️⃣ **Câu 21: Hãy mô tả chi tiết data flow từ lúc user gửi query đến lúc nhận response.**

**Trả lời:**

```
┌─ User Interface (Gradio) ─┐
│   User Input: "Câu hỏi?"   │
└───────────┬────────────────┘
            │
            ▼
┌─ get_response() in app.py ─────────────────┐
│ 1. Validate app_config, app_db            │
│ 2. Print debug info                        │
└──────────┬─────────────────────────────────┘
           │
           ▼
┌─ retrieve_documents() ─────────────────────┐
│ 1. Expand query (thêm alias)              │
│ 2. Create MMR retriever                    │
│ 3. Query ChromaDB: fetch_k=20, k=10       │
│ 4. Return: List[RetrievedContext]         │
└──────────┬─────────────────────────────────┘
           │
           ▼
┌─ ChromaDB ─────────────────────────────────┐
│ 1. Vector search (cosine similarity)       │
│ 2. Fetch top-20 candidates                │
│ 3. MMR algorithm selects top-10            │
│ 4. Return: List[Document]                 │
└──────────┬─────────────────────────────────┘
           │
           ▼
┌─ create_prompt() ──────────────────────────┐
│ 1. Apply context budget (max 6000 chars)  │
│ 2. Format each context with citation ID    │
│ 3. Concatenate contexts                    │
│ 4. Substitute into template                │
│    {context} → formatted contexts          │
│    {question} → user query                 │
│ 5. Return: Prompt string                   │
└──────────┬─────────────────────────────────┘
           │
           ▼
┌─ _prepare_messages() ──────────────────────┐
│ 1. Convert history to standard format      │
│ 2. Add current prompt as user message      │
│ 3. Return: List[{role, content}]          │
└──────────┬─────────────────────────────────┘
           │
           ▼
┌─ call_llm_api() ───────────────────────────┐
│ 1. Check provider (openrouter)             │
│ 2. Call _call_openrouter()                 │
└──────────┬─────────────────────────────────┘
           │
           ▼
┌─ OpenRouter API ───────────────────────────┐
│ 1. Receive: POST /v1/chat/completions      │
│ 2. Model: openrouter/owl-alpha (or other) │
│ 3. Process: Generate response              │
│ 4. Return: {"choices": [...]}              │
└──────────┬─────────────────────────────────┘
           │
           ▼
┌─ Parse Response ───────────────────────────┐
│ 1. Extract: result["choices"][0]["message"]│
│ 2. Get: message["content"]                 │
│ 3. Log timing info                         │
│ 4. Return: Answer string                   │
└──────────┬─────────────────────────────────┘
           │
           ▼
┌─ Gradio UI ────────────────────────────────┐
│ Display: [Response Text]                   │
│ Add to history: (question, answer)         │
└────────────────────────────────────────────┘
```

---

### 2️⃣2️⃣ **Câu 22: Giải thích chiến lược context budgeting trong create_prompt().**

**Trả lời:**

**Hàm _apply_context_budget():**
```python
max_chars = 6000  # Tổng độ dài max
max_chars_per_chunk = 1500  # Max per document
```

**Mục đích:**
1. **Token limit:** LLM có giới hạn context window
2. **Cost optimization:** OpenRouter tính phí theo tokens
3. **Relevance:** Context dài quá sẽ làm giảm chất lượng

**Algorithm:**
1. Duyệt qua contexts lần lượt
2. Cắt ngắn mỗi context nếu > 1500 chars
3. Format context (thêm citation ID, source, chunk ID)
4. Cộng dồn độ dài
5. Stop khi tổng > 6000 chars hoặc hết contexts
6. Return: Pruned contexts

---

### 2️⃣3️⃣ **Câu 23: Hàm _expand_query() có tác dụng gì?**

**Trả lời:**

```python
def _expand_query(query, enabled=True):
    """Alias expansion for Vietnamese acronyms"""
    if "dhqghn" in query.lower():
        aliases = ["ĐHQGHN", "Đại học Quốc gia Hà Nội"]
    return f"{query}\n" + "\n".join(aliases)
```

**Tác dụng:**
- Expand Vietnamese acronyms thành full name
- Ví dụ: "DHQGHN" → "DHQGHN\nĐHQGHN\nĐại học Quốc gia Hà Nội"
- Giúp retrieval tìm được documents dùng từ khác nhau
- Cải thiện recall (coverage) của retriever

---

### 2️⃣4️⃣ **Câu 24: Load_chroma_db() làm gì? Tại sao cần load vào bộ nhớ?**

**Trả lời:**

```python
def load_chroma_db(config):
    db_path = config["db_dir"]
    embeddings_model = config["api"]["huggingface"]["embedding_model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    embeddings = HuggingFaceEmbeddings(
        model_name=embeddings_model,
        model_kwargs={"device": device}
    )
    
    db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )
    return db
```

**Tại sao load vào bộ nhớ:**
- **Performance:** Query nhanh hơn (không phải load từ disk mỗi lần)
- **Reusability:** Dùng chung db instance cho tất cả requests
- **GPU support:** Load embedding model lên GPU nếu có

---

### 2️⃣5️⃣ **Câu 25: Các lỗi tiềm ẩn trong code là gì? Làm sao fix?**

**Trả lời:**

**Lỗi 1: KeyError trong config_loader.py**
- **Problem:** Kiểm tra `"huggingface_api_token"` ở cấp ngoài, nhưng thực tế nó nằm trong `config["api"]["huggingface"]`
- **Fix:** Sửa logic kiểm tra để access đúng vị trí lồng nhau

**Lỗi 2: OpenRouter 403 Forbidden**
- **Problem:** API key sai, hết credits, hoặc endpoint sai
- **Fix:** Kiểm tra API key, nạp credits, đảm bảo endpoint đúng

**Lỗi 3: ChromaDB không tìm thấy context**
- **Problem:** Chưa chạy `create_chroma_db.py`
- **Fix:** Chạy script tạo database trước

**Lỗi 4: Symlinks warning trên Windows**
- **Problem:** HuggingFace không support symlinks trên Windows
- **Fix:** Disable warning bằng: `os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"`

---

### 2️⃣6️⃣ **Câu 26: Hãy thiết kế cải tiến cho hệ thống này.**

**Trả lời:**

**Cải tiến 1: Multi-stage Retrieval**
- **Problem:** Single-pass retrieval có thể miss relevant documents
- **Solution:** Implement 2-stage retrieval
  - Stage 1: BM25 (keyword) + Dense (semantic)
  - Stage 2: Re-ranking bằng LLM hoặc cross-encoder
- **Benefit:** Cải thiện recall & precision

**Cải tiến 2: Query Decomposition**
- **Problem:** Complex queries không được tách ra
- **Solution:** Dùng LLM để tách complex query thành sub-queries
- **Benefit:** Tăng accuracy cho câu hỏi phức tạp

**Cải tiến 3: Caching**
- **Problem:** Query giống nhau phải retrieve & call LLM lại
- **Solution:** Cache results bằng Redis hoặc memory cache
- **Benefit:** Giảm latency, tiết kiệm chi phí API

**Cải tiến 4: Streaming Response**
- **Problem:** Phải đợi LLM trả lời xong mới hiển thị
- **Solution:** Implement streaming chunks
- **Benefit:** UX tốt hơn, cảm giác response nhanh hơn

**Cải tiến 5: Feedback Loop**
- **Problem:** Không có feedback từ user
- **Solution:** Thêm thumbs up/down, allow user annotation
- **Benefit:** Continuous improvement bằng RLHF

---

### 2️⃣7️⃣ **Câu 27: Hãy giải thích MMR algorithm chi tiết.**

**Trả lời:**

**MMR (Maximal Marginal Relevance) Algorithm:**

```
Input: Query q, Documents D, fetch_k, k
Output: Top-k documents

1. Initial retrieval:
   - Tính similarity(q, d) cho mỗi document d ∈ D
   - Sắp xếp theo similarity, lấy top fetch_k documents
   
2. Iterative selection:
   - R = {} (selected documents)
   - C = top-fetch_k documents (candidates)
   
   FOR i = 1 to k:
       - FOR mỗi document d ∈ C:
           - relevance = similarity(q, d)  # Độ liên quan với query
           - redundancy = max_j∈R similarity(d, j)  # Độ lặp với selected docs
           - mmr_score(d) = λ * relevance - (1-λ) * redundancy
       
       - d_best = argmax MMR_score
       - R.add(d_best)
       - C.remove(d_best)
   
3. Return R (top-k diverse documents)
```

**Parameters:**
- `λ (lambda)` = balance factor (default: 0.5)
  - λ = 1: Pure relevance (= simple similarity search)
  - λ = 0: Pure diversity
  - λ = 0.5: Balance relevance & diversity

**Lợi ích:**
- Tránh documents redundant
- Đảm bảo tính đa dạng
- Cải thiện coverage (coverage) của information

---

### 2️⃣8️⃣ **Câu 28: So sánh ChromaDB với các vector database khác (Pinecone, Weaviate, Milvus).**

**Trả lời:**

| Tính chất | ChromaDB | Pinecone | Weaviate | Milvus |
|----------|----------|----------|----------|--------|
| **Setup** | Local/embedded | Cloud | Cloud/self-hosted | Self-hosted |
| **Cost** | Free | Paid | Freemium | Free |
| **Persistence** | SQLite | Managed | Managed | Customizable |
| **Scaling** | Limited | Auto | Manual | Manual |
| **Latency** | ms (local) | ms (network) | ms (network) | μs-ms |
| **Use case** | Development/small | Production/large | Production | Production/large-scale |
| **LangChain integration** | ✅ Official | ✅ | ✅ | ✅ |
| **Learning curve** | Easy | Easy | Medium | Hard |

**Chọn ChromaDB vì:**
- Dự án là development/POC → không cần production setup
- Local storage → privacy, không cần internet
- Easy integration với LangChain → nhanh prototyping
- Free & open-source

---

### 2️⃣9️⃣ **Câu 29: Giải thích cách xử lý lỗi HTTP trong _call_openrouter().**

**Trả lời:**

```python
try:
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read().decode("utf-8")
except urllib.error.HTTPError as exc:
    # Handle HTTP errors (4xx, 5xx)
    error_body = exc.read().decode("utf-8")
    raise RuntimeError(
        f"OpenRouter request failed: {exc.code} {exc.reason} - {error_body}"
    )

# Parse response
result = json.loads(raw)
if not result or "choices" not in result or len(result["choices"]) == 0:
    raise ValueError("No valid response from OpenRouter")
```

**Error cases:**
1. **403 Forbidden:** API key sai, hết credits
2. **429 Rate Limited:** Quá nhiều requests
3. **500 Server Error:** OpenRouter server down
4. **Timeout:** Network chậm

**Best practice:**
- Catch HTTPError riêng để lấy error body
- Validate response structure trước parse JSON
- Set timeout để tránh hang

---

### 3️⃣0️⃣ **Câu 30: Đề xuất solution architecture cho việc scale hệ thống này lên production.**

**Trả lời:**

```
┌─────────────────────────────────────────────────────────────┐
│                   PRODUCTION ARCHITECTURE                    │
└─────────────────────────────────────────────────────────────┘

Frontend Layer:
├─ React/Vue.js Web UI
├─ Mobile App (iOS/Android)
└─ API Gateway (FastAPI/Flask)

Service Layer:
├─ API Service (FastAPI)
│  └─ Endpoint: POST /chat, GET /history
├─ Retriever Service (Python)
│  └─ ChromaDB or Milvus backend
├─ LLM Service (Wrapper)
│  └─ OpenRouter or self-hosted LLM
└─ Cache Service (Redis)
   └─ Query cache, response cache

Data Layer:
├─ Vector DB: Milvus (production-grade)
├─ SQL DB: PostgreSQL (metadata, history)
├─ Cache: Redis (ephemeral data)
└─ Object Storage: S3 (documents, embeddings)

Infrastructure:
├─ Kubernetes orchestration
├─ Docker containers
├─ Load balancer (Nginx)
├─ Message queue (RabbitMQ)
└─ Monitoring (Prometheus + Grafana)

ML Pipeline:
├─ Data ingestion (ETL)
├─ Embedding pipeline (batch)
├─ Embedding update scheduler (daily/weekly)
└─ Model evaluation (BLEU, ROUGE metrics)
```

**Key improvements:**
1. **Scalability:** Horizontal scaling bằng Kubernetes
2. **Reliability:** Load balancing, failover
3. **Performance:** Caching, CDN, edge computing
4. **Monitoring:** Metrics, alerting
5. **Data management:** Proper DB, backup strategy
6. **Security:** Auth, HTTPS, API keys management
7. **Cost:** Auto-scaling, resource optimization

---

## 📚 Tài Liệu Tham Khảo

- **LangChain Docs:** https://python.langchain.com/docs/
- **ChromaDB Docs:** https://docs.trychroma.com/
- **OpenRouter Docs:** https://openrouter.ai/docs
- **HuggingFace Embedding Models:** https://huggingface.co/models?pipeline_tag=feature-extraction
- **Gradio Documentation:** https://www.gradio.app/docs/

---

## 💡 Mẹo Phỏng Vấn

✅ **DO:**
- Hiểu rõ kiến trúc tổng thể trước khi đi vào chi tiết
- Giải thích tại sao lựa chọn tech stack này
- Có thể vẽ sơ đồ khi giải thích
- Nêu trade-off của từng lựa chọn
- Đề xuất cải tiến và giải thích lợi ích

❌ **DON'T:**
- Chỉ đọc code mà không hiểu ý nghĩa
- Không biết tại sao dùng MMR search
- Không thể giải thích error handling
- Không có ý kiến về cải tiến
- Nói quá nhanh không cho phỏng vấn hỏi lại

---

## 🎯 Chuẩn Bị Cuối Cùng

**Trước phỏng vấn:**
1. ✅ Clone repo, chạy setup local
2. ✅ Run create_chroma_db.py
3. ✅ Test app.py, thử vài câu hỏi
4. ✅ Đọc kỹ README.md và project_architecture.md
5. ✅ Prepare mini-demo hoặc recorded demo

**Trong phỏng vấn:**
1. 🎤 Giải thích high-level architecture trước
2. 🎤 Drill down vào technical details khi được hỏi
3. 🎤 Sử dụng whiteboard/paper để vẽ sơ đồ
4. 🎤 Hỏi lại clarifying questions nếu không hiểu
5. 🎤 Giải thích code bằng code examples, không chỉ miệng

---

**Chúc bạn phỏng vấn thành công! 🚀**
