import numpy as np
import pandas as pd
import sys,os
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.decorators import handle_exception

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifacts","model.pkl")
    model_history_path = os.path.join("artifacts","model_history","model_history.csv")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    @handle_exception
    def initiate_model_traning(self, train_arr, test_arr):
        """
            This is function to Train and find best model from choosen models.
        """
        pass


