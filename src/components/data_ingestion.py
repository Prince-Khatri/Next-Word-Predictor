import os
import random
from typing import List, Tuple

from src.decorators import handle_exception
from src.logger import logging
from src.utils import load_yaml


class DataIngestion:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_yaml(config_path)
        self.artifacts_cfg = self.config["artifacts"]
        self.data_cfg = self.config["data"]

    @staticmethod
    def _load_text(file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    @staticmethod
    def _save_lines(file_path: str, lines: List[str]) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as file:
            file.write("\n".join(lines))

    @staticmethod
    def _split_data(
        data: List[str],
        train_ratio: float,
        val_ratio: float,
        seed: int,
    ) -> Tuple[List[str], List[str], List[str]]:
        random.seed(seed)
        random.shuffle(data)
        n = len(data)
        train_end = int(train_ratio * n)
        val_end = int((train_ratio + val_ratio) * n)
        return data[:train_end], data[train_end:val_end], data[val_end:]

    @handle_exception
    def initiate_data_ingestion(self, sentences: List[str]) -> dict:
        logging.info("Starting data ingestion and split.")
        train_data, val_data, test_data = self._split_data(
            data=sentences.copy(),
            train_ratio=self.data_cfg["train_split"],
            val_ratio=self.data_cfg["val_split"],
            seed=self.data_cfg["random_seed"],
        )

        self._save_lines(self.artifacts_cfg["train_sentences_path"], train_data)
        self._save_lines(self.artifacts_cfg["val_sentences_path"], val_data)
        self._save_lines(self.artifacts_cfg["test_sentences_path"], test_data)

        logging.info(
            "Data split complete. Train=%s, Val=%s, Test=%s",
            len(train_data),
            len(val_data),
            len(test_data),
        )
        return {
            "train_sentences": train_data,
            "val_sentences": val_data,
            "test_sentences": test_data,
            "train_path": self.artifacts_cfg["train_sentences_path"],
            "val_path": self.artifacts_cfg["val_sentences_path"],
            "test_path": self.artifacts_cfg["test_sentences_path"],
        }
