import os
from typing import Any, Dict

import yaml


def _load_env_file(dotenv_path: str = ".env") -> None:
    """Đọc file .env nếu tồn tại và nạp các biến vào os.environ"""
    if os.path.exists(dotenv_path):
        try:
            with open(dotenv_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # Bỏ qua dòng trống hoặc comment
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip()
                        # Loại bỏ dấu ngoặc kép hoặc ngoặc đơn ở đầu/cuối nếu có
                        if len(val) >= 2 and (
                            (val.startswith('"') and val.endswith('"')) or
                            (val.startswith("'") and val.endswith("'"))
                        ):
                            val = val[1:-1]
                        # Chỉ đặt nếu chưa được đặt trong môi trường
                        os.environ.setdefault(key, val)
        except Exception:
            pass


def load_config(config_path: str = "config.yaml") -> Dict[Any, Any]:
    """Load config from yaml file

    Args:
        config_path: Đường dẫn đến file config

    Returns:
        Dict chứa config
    """
    # Nạp các biến môi trường từ file .env trước khi đọc config
    _load_env_file()

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    api_config = config.get("api", {})

    hf_config = api_config.get("huggingface", {})
    if isinstance(hf_config, dict):
        hf_config["token"] = os.environ.get(
            "HUGGINGFACEHUB_API_TOKEN",
            os.environ.get("HF_TOKEN", hf_config.get("token", "")),
        )

    openrouter_config = api_config.get("openrouter", {})
    if isinstance(openrouter_config, dict):
        openrouter_config["token"] = os.environ.get(
            "OPENROUTER_API_KEY", openrouter_config.get("token", "")
        )

    return config
