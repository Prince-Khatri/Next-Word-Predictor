import re
from typing import Dict, List

from tensorflow.keras.preprocessing.text import Tokenizer

from src.decorators import handle_exception
from src.logger import logging
from src.utils import load_yaml, save_object


class DataTransformation:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_yaml(config_path)
        self.data_cfg = self.config["data"]
        self.model_cfg = self.config["model"]
        self.artifacts_cfg = self.config["artifacts"]

    @staticmethod
    def preprocess_text(text: str, min_words: int) -> List[str]:
        text = text.lower()
        text = re.sub(r"=+[^=]+=+", " ", text)
        text = re.sub(r"<unk>", " ", text)
        text = re.sub(r"@-@", "-", text)
        text = re.sub(r"[^a-z0-9.,!?\'\"\-\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if len(s.split()) >= min_words]

    @handle_exception
    def tokenize_data(
        self,
        train_sentences: List[str],
        val_sentences: List[str],
        test_sentences: List[str],
    ) -> Dict:
        logging.info("Fitting tokenizer on train data.")
        tokenizer = Tokenizer(
            num_words=self.model_cfg["max_vocab"],
            oov_token="<OOV>",
        )
        tokenizer.fit_on_texts(train_sentences)

        total_words = min(self.model_cfg["max_vocab"], len(tokenizer.word_index)) + 1
        train_seq = tokenizer.texts_to_sequences(train_sentences)
        val_seq = tokenizer.texts_to_sequences(val_sentences)
        test_seq = tokenizer.texts_to_sequences(test_sentences)

        save_object(self.artifacts_cfg["tokenizer_path"], tokenizer)
        logging.info("Tokenizer saved at %s", self.artifacts_cfg["tokenizer_path"])

        return {
            "tokenizer": tokenizer,
            "total_words": total_words,
            "train_seq": train_seq,
            "val_seq": val_seq,
            "test_seq": test_seq,
        }
