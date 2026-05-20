import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gradio as gr

from src import load_chroma_db, load_config, retrieve_documents
from src.llm import call_llm_api, create_prompt


app_config = None
app_db = None


def get_response(message: str, history=None) -> str:
    """Handle a chat question with the RAG pipeline."""
    global app_config, app_db

    try:
        if not app_config or not app_db:
            return "Loi: Tai nguyen chua duoc tai. Hay khoi dong lai ung dung."

        print(f"Truy van: '{message}'")
        started = time.perf_counter()
        contexts = retrieve_documents(app_db, message, app_config["retriever_k"], config=app_config)
        retrieved_at = time.perf_counter()

        if not contexts:
            return "I do not have enough information from the provided context."

        debug_enabled = bool(app_config.get("debug")) or os.environ.get("RAG_DEBUG") == "1"
        if debug_enabled:
            print(f"Retrieved {len(contexts)} contexts:")
            for c in contexts:
                title = c.get("title", "")
                url = c.get("url", "")
                chunk_id = c.get("chunk_id", "")
                source = c.get("source", "")
                print(f"- [{c.get('citation_id')}] title={title} url={url} chunk_id={chunk_id} source={source}")

        prompt = create_prompt(message, contexts, app_config["template"], config=app_config)
        answer = call_llm_api(prompt, app_config, history)
        finished_at = time.perf_counter()
        print(
            "Timing: "
            f"retrieval={retrieved_at - started:.2f}s, "
            f"llm={finished_at - retrieved_at:.2f}s, "
            f"total={finished_at - started:.2f}s"
        )
        return answer

    except Exception as e:
        print(f"Loi: {str(e)}")
        return f"Xin loi, co loi xay ra: {str(e)}"


def initialize_app() -> bool:
    """Initialize config and Chroma database once at startup."""
    global app_config, app_db

    try:
        app_config = load_config()
        if not app_config:
            print("Loi: Khong the doc file config.yaml")
            return False

        print("Da tai config:")
        provider = app_config.get("api", {}).get("provider", "openrouter")
        provider_cfg = app_config.get("api", {}).get(provider, {})
        model_id = provider_cfg.get("model_id", "unknown")
        embedding_model = (
            app_config.get("api", {}).get("huggingface", {}).get("embedding_model", "unknown")
        )
        print(f"- Provider: {provider}")
        print(f"- Model: {model_id}")
        print(f"- Embedding: {embedding_model}")

        app_db = load_chroma_db(app_config)
        return True

    except Exception as e:
        print(f"Loi khoi tao: {str(e)}")
        return False


examples = [
    "What bank, which is the 5th largest in the US, is based in Pittsburgh?",
    "How many neighborhoods does Pittsburgh have?",
    "Who is Pittsburgh named after?",
    "What famous vaccine was developed at University of Pittsburgh in 1955?",
    "DHQGHN duoc thanh lap khi nao?",
    "VNU-USSH duoc thanh lap khi nao",
]

demo = gr.ChatInterface(
    fn=get_response,
    title="RAG Chatbot ve Pittsburgh / CMU va VNU",
    description="Chatbot co kha nang truy van du lieu ve Pittsburgh / CMU va VNU",
    examples=examples,
)


if __name__ == "__main__":
    print("Dang khoi dong RAG chatbot...")

    if initialize_app():
        demo.launch()
    else:
        print("Khong the khoi dong ung dung do loi khoi tao")
        exit(1)
