import importlib.util
from pathlib import Path


def load_module(module_name: str, relative_path: str):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_create_prompt_keeps_context_boundaries_and_citations():
    api = load_module("llm_api", "src/llm/api.py")
    template = "Information:\n{context}\n\nQuestion: {question}\nAnswer:"
    contexts = [
        {
            "citation_id": 1,
            "title": "Source A",
            "url": "https://example.com/a",
            "chunk_id": 2,
            "score": 0.91,
            "content": "First chunk.",
        },
        {
            "citation_id": 2,
            "title": "Source B",
            "source": "local.json",
            "chunk_id": 3,
            "content": "Second chunk.",
        },
    ]

    prompt = api.create_prompt("What is included?", contexts, template)

    assert "[1] Source A | score=0.9100" in prompt
    assert "Source: https://example.com/a" in prompt
    assert "\n\n---\n\n" in prompt
    assert "[2] Source B" in prompt
    assert "Question: What is included?" in prompt

