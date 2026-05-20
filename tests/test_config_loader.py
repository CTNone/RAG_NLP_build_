import importlib.util
from pathlib import Path


def load_module(module_name: str, relative_path: str):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_config_prefers_environment_tokens(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
api:
  provider: openrouter
  huggingface:
    token: file-hf-token
    embedding_model: test-embedding
  openrouter:
    token: file-openrouter-token
    model_id: test-model
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "env-hf-token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-openrouter-token")

    config_loader = load_module("config_loader", "src/utils/config_loader.py")
    config = config_loader.load_config(str(config_path))

    assert config["api"]["huggingface"]["token"] == "env-hf-token"
    assert config["api"]["openrouter"]["token"] == "env-openrouter-token"

