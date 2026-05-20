import importlib.util
from pathlib import Path


def load_module(module_name: str, relative_path: str):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eval_match_substring_and_regex():
    matching = load_module("matching", "src/eval/matching.py")
    match_answer = matching.match_answer

    assert match_answer("It was founded in 1993.", "1993", "") is True
    assert match_answer("It was founded in 1993.", "1994", "") is False
    assert (
        match_answer("10 thang 12 nam 1993", "", r"\b10\s+thang\s+12\s+nam\s+1993\b")
        is True
    )
    assert match_answer("10 thang 12 nam 1993", "", r"(") is False
    assert match_answer("anything", "", "") is None
