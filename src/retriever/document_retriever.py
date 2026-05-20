import json
import os
import re
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

RetrievedContext = Dict[str, Any]


def _expand_query(query: str, enabled: bool = True) -> str:
    """Very small alias expansion for common unaccented Vietnamese acronyms."""
    if not enabled:
        return query

    normalized = query.lower()
    aliases: List[str] = []

    if "dhqghn" in normalized:
        aliases.extend(["ĐHQGHN", "Đại học Quốc gia Hà Nội"])

    if not aliases:
        return query

    uniq = list(dict.fromkeys(aliases))
    return f"{query}\n" + "\n".join(uniq)


def _build_context(doc: Document, citation_id: int) -> RetrievedContext:
    metadata = doc.metadata or {}
    return {
        "citation_id": citation_id,
        "content": doc.page_content,
        "source": metadata.get("source", ""),
        "url": metadata.get("url", ""),
        "title": metadata.get("title", ""),
        "section": metadata.get("section", ""),
        "chunk_id": metadata.get("chunk_id", ""),
    }


def retrieve_documents(db, query: str, k: int = 3, config: Optional[dict] = None) -> List[RetrievedContext]:
    """Single-path retrieval: vector MMR, then take top-k."""
    config = config or {}
    fetch_k = int(config.get("retriever_fetch_k", max(20, k)))
    fetch_k = max(fetch_k, k)

    expanded_query = _expand_query(query, enabled=True)
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k},
    )
    docs: List[Document] = retriever.invoke(expanded_query)
    return [_build_context(doc, i + 1) for i, doc in enumerate(docs)]


def load_json_documents(file_path: str) -> List[Document]:
    """Load segmented JSON and convert to Documents with normalized metadata."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "chunks" not in data or not isinstance(data["chunks"], list):
            return []

        url = data.get("url", "") or ""
        title = data.get("title", "") or os.path.basename(file_path)

        docs: List[Document] = []
        seen_hashes = set()

        max_chunk_chars = 4000
        for i, chunk in enumerate(data["chunks"]):
            chunk_text = (chunk or "").strip()
            if not chunk_text:
                continue
            if len(chunk_text) < 20:
                continue
            if len(chunk_text) > max_chunk_chars:
                chunk_text = chunk_text[:max_chunk_chars]

            h = hash(chunk_text)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            section = ""
            for line in chunk_text.splitlines()[:20]:
                m = re.match(r"^\s*#{1,6}\s+(.+?)\s*$", line)
                if m:
                    section = m.group(1)[:200]
                    break

            docs.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "source": file_path,
                        "url": url,
                        "title": title,
                        "section": section,
                        "chunk_id": i,
                    },
                )
            )

        return docs
    except Exception:
        return []


def get_data_directories(base_path: str, prefix: str) -> List[str]:
    """Auto-discover dataset directories in base_path by prefix."""
    if not os.path.exists(base_path):
        return []

    return [
        os.path.join(base_path, item)
        for item in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, item)) and item.startswith(prefix)
    ]

