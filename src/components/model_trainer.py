import os
from typing import Dict, List

import numpy as np
import math
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.decorators import handle_exception
from src.logger import logging
from src.utils import load_yaml, ensure_parent_dir


class ModelTrainer:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_yaml(config_path)
        self.model_cfg = self.config["model"]
        self.artifacts_cfg = self.config["artifacts"]

    @staticmethod
    def data_generator(sequences: List[List[int]], max_len: int, batch_size: int):
        while True:
            x_batch, y_batch = [], []
            for seq in sequences:
                for i in range(1, len(seq)):
                    n_gram = seq[max(0, i - max_len) : i + 1]
                    n_gram = pad_sequences([n_gram], maxlen=max_len, padding="pre")[0]
                    x_batch.append(n_gram[:-1])
                    y_batch.append(n_gram[-1])
                    if len(x_batch) == batch_size:
                        yield np.array(x_batch), np.array(y_batch)
                        x_batch, y_batch = [], []

    def _count_samples(self, sequences: List[List[int]], max_len: int) -> int:
        return sum(max(len(seq) - max_len, 0) for seq in sequences)

    def build_model(self, total_words: int, model_path: str) -> tf.keras.Model:
        # if model exist and flag tells to continue traning continue traning
        resume = self.config['model'].get('resume_training', False)
        if resume and os.path.exists(model_path) :
            logging.info('Resuming the model traning')
            model = tf.keras.models.load_model(model_path)
        else:
            logging.info('Building fresh model.')
            max_len = self.model_cfg["max_len"]
            model = Sequential(
                [
                    Embedding(total_words, self.model_cfg["embedding_dim"], input_length=max_len - 1),
                    LSTM(self.model_cfg["lstm_units"], return_sequences=True),
                    Dropout(self.model_cfg["dropout_rate"]),
                    LSTM(self.model_cfg["lstm_units"]),
                    Dense(total_words, activation="softmax"),
                ]
            )
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer='adam',
            metrics=["accuracy"],
        )
        return model

    @handle_exception
    def train(self, train_seq: List[List[int]], val_seq: List[List[int]], total_words: int) -> Dict:
        
        base_path = self.config["artifacts"]["model_dir"]
        
        model_path = base_path + '/' + self.config["artifacts"]["model_file_name"]
        ensure_parent_dir(model_path)
        history_path = base_path + '/' + self.config["artifacts"]["history_file_name"]
        ensure_parent_dir(history_path)

        model = self.build_model(total_words=total_words, model_path=model_path)
        max_len = self.model_cfg["max_len"]
        batch_size = self.model_cfg["batch_size"]

        train_samples = self._count_samples(train_seq, max_len)
        val_samples = self._count_samples(val_seq, max_len)
        steps_per_epoch = max(1, train_samples // batch_size)
        validation_steps = max(1, val_samples // batch_size)

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.model_cfg["early_stopping_patience"],
                restore_best_weights=True,
            )
        ]

        logging.info("Starting model training.")
        history = model.fit(
            self.data_generator(train_seq, max_len, batch_size),
            steps_per_epoch=steps_per_epoch,
            epochs=self.model_cfg["epochs"],
            validation_data=self.data_generator(val_seq, max_len, batch_size),
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1,
        )

        model.save(model_path)
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(history_path, index=False)
        logging.info("Model and history saved.")

        final_loss = float(history.history["val_loss"][-1])
        final_acc = float(history.history["val_accuracy"][-1])
        perplexity = float(np.exp(final_loss))
        return {
            "model": model,
            "history": history.history,
            "val_loss": final_loss,
            "val_accuracy": final_acc,
            "perplexity": perplexity,
        }

    @handle_exception
    def evaluate(self, model: tf.keras.Model, test_seq: List[List[int]]) -> Dict[str, float]:
        max_len = self.model_cfg["max_len"]
        batch_size = self.model_cfg["batch_size"]
        test_samples = self._count_samples(test_seq, max_len)
        test_steps = math.ceil(test_samples / batch_size)
        loss, accuracy = model.evaluate(
            self.data_generator(test_seq, max_len, batch_size),
            steps=test_steps,
            verbose=1,
        )
        return {"test_loss": float(loss), "test_accuracy": float(accuracy)}
