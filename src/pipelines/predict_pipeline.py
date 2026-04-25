import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.utils import load_object, load_yaml


class PredictPipeline:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_yaml(config_path)
        self.model_cfg = self.config["model"]
        self.inference_cfg = self.config["inference"]
        self.artifacts_cfg = self.config["artifacts"]

        self.model = tf.keras.models.load_model(self.artifacts_cfg["best_model_path"], compile=False)
        self.tokenizer = load_object(self.artifacts_cfg["tokenizer_path"])
        self.index_word = {v: k for k, v in self.tokenizer.word_index.items()}

    def predict_next_words(
        self,
        seed_text: str,
        next_words: int | None = None,
        top_k: int | None = None,
    ) -> str:
        next_words = next_words or self.inference_cfg["default_next_words"]
        top_k = top_k or self.inference_cfg["top_k"]

        result = seed_text.strip()
        max_len = self.model_cfg["max_len"]

        for _ in range(next_words):
            token_seq = self.tokenizer.texts_to_sequences([result])[0]
            token_seq = pad_sequences([token_seq], maxlen=max_len - 1, padding="pre")
            probs = self.model.predict(token_seq, verbose=0)[0]

            oov_idx = self.tokenizer.word_index.get("<OOV>", 1)
            if oov_idx < len(probs):
                probs[oov_idx] = 0.0

            k = min(top_k, len(probs))
            top_indices = np.argsort(probs)[-k:]
            top_probs = probs[top_indices]

            if top_probs.sum() == 0:
                break

            top_probs = top_probs / top_probs.sum()
            predicted_idx = int(np.random.choice(top_indices, p=top_probs))
            next_word = self.index_word.get(predicted_idx, "")
            if not next_word:
                break
            result = f"{result} {next_word}".strip()

        return result
