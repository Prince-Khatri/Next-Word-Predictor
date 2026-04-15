import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.decorators import handle_exception


@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    @handle_exception
    def get_data_preprocessor(self):
        """
            This Function is responsible for getting preprocessor object.
        """
        pass

    @handle_exception   
    def initiate_data_transformation(self, train_path, test_path):
        """
            This function uses preprocessor to transform train and test set.
        """
        pass
