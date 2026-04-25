import math

import mlflow

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.utils import load_yaml, ensure_parent_dir


class TrainPipeline:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_yaml(config_path)
        self.config_path = config_path

    def run(self) -> dict:
        data_ingestion = DataIngestion(config_path=self.config_path)
        data_transformation = DataTransformation(config_path=self.config_path)
        model_trainer = ModelTrainer(config_path=self.config_path)

        raw_text_path = self.config["data"]["raw_text_path"]
        with open(raw_text_path, "r", encoding="utf-8") as file:
            text = file.read()

        sentences = data_transformation.preprocess_text(
            text=text,
            min_words=self.config["data"]["min_words_per_sentence"],
        )
        split_data = data_ingestion.initiate_data_ingestion(sentences)
        token_data = data_transformation.tokenize_data(
            train_sentences=split_data["train_sentences"],
            val_sentences=split_data["val_sentences"],
            test_sentences=split_data["test_sentences"],
        )

        mlflow_cfg = self.config["mlflow"]
        if mlflow_cfg["tracking_uri"]:
            mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
        mlflow.set_experiment(mlflow_cfg["experiment_name"])

        with mlflow.start_run(run_name=self.config['mlflow']['run_name']):
            mlflow.set_tags(
                {
                    'vocab_size': self.config['mlflow']['tags']['vocab_size'],
                    'performance': self.config['mlflow']['tags']['performance'],
                    'data_set_version': self.config['mlflow']['tags']['data_set']
                }
            )
            mlflow.log_params(
                {
                    "vocab_size": token_data["total_words"],
                    "sequence_length": self.config["model"]["max_len"],
                    "embedding_dim": self.config["model"]["embedding_dim"],
                    "lstm_units": self.config["model"]["lstm_units"],
                    "dropout_rate": self.config["model"]["dropout_rate"],
                    "batch_size": self.config["model"]["batch_size"],
                    "epochs": self.config["model"]["epochs"],
                }
            )

            train_out = model_trainer.train(
                train_seq=token_data["train_seq"],
                val_seq=token_data["val_seq"],
                total_words=token_data["total_words"],
            )
            eval_out = model_trainer.evaluate(
                model=train_out["model"],
                test_seq=token_data["test_seq"],
            )

            mlflow.log_metrics(
                {
                    "loss": train_out["val_loss"],
                    "accuracy": train_out["val_accuracy"],
                    "perplexity": train_out["perplexity"],
                    "test_loss": eval_out["test_loss"],
                    "test_accuracy": eval_out["test_accuracy"],
                }
            )

            base_path = self.config["artifacts"]["model_dir"]
            model_path = base_path + '/' + self.config["artifacts"]["model_file_name"]
            history_path = base_path + '/' + self.config["artifacts"]["history_file_name"]

            ensure_parent_dir(base_path)
            mlflow.log_artifact(
                model_path,
                artifact_path="model"
            )

            mlflow.log_artifact(
                self.config["artifacts"]["tokenizer_path"],
                artifact_path="tokenizer"
            )

            mlflow.log_artifact(
                history_path,
                artifact_path="training"
            )

            logging.info(
                "Training complete. Loss=%.4f Accuracy=%.4f Perplexity=%.4f",
                train_out["val_loss"],
                train_out["val_accuracy"],
                train_out["perplexity"],
            )

            return {
                "loss": train_out["val_loss"],
                "accuracy": train_out["val_accuracy"],
                "perplexity": train_out["perplexity"],
                "test_loss": eval_out["test_loss"],
                "test_accuracy": eval_out["test_accuracy"],
                "is_finite_perplexity": math.isfinite(train_out["perplexity"]),
            }


if __name__ == "__main__":
    output = TrainPipeline().run()
    print(output)
