import os
from typing import Any, Dict

import yaml


def load_config(config_path: str = "config.yaml") -> Dict[Any, Any]:
    """Load config from yaml file

    Args:
        config_path: Đường dẫn đến file config

    Returns:
        Dict chứa config
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Xử lý biến môi trường
    if "huggingface_api_token" in config:
        config["huggingface_api_token"] = os.environ.get(
            "HUGGINGFACEHUB_API_TOKEN", config["huggingface_api_token"]
        )

    return config
