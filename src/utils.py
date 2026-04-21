import os
from typing import Any

import dill
import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def save_object(path: str, obj: Any) -> None:
    ensure_parent_dir(path)
    with open(path, "wb") as file:
        dill.dump(obj, file)


def load_object(path: str) -> Any:
    with open(path, "rb") as file:
        return dill.load(file)
