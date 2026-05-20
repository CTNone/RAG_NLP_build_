import importlib.util
from pathlib import Path


def load_module(module_name: str, relative_path: str):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expand_query_adds_vnu_aliases_for_unaccented_acronym():
    retriever = load_module("document_retriever", "src/retriever/document_retriever.py")

    expanded = retriever._expand_query("DHQGHN duoc thanh lap khi nao?")

    assert "DHQGHN duoc thanh lap khi nao?" in expanded
    assert "\u0110HQGHN" in expanded
    assert "\u0110\u1ea1i h\u1ecdc Qu\u1ed1c gia H\u00e0 N\u1ed9i" in expanded
    # Keep expansion minimal for speed (no extra keywords unless proven needed)
