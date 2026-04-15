from setuptools import setup, find_packages
from typing import List
import src.constants as constants

REQUIREMENTS_FILE_PATH = constants.REQUIREMENTS_FILE_PATH

def get_requirements(file_path:str)->List[str]:
    with open(file_path, 'r') as f:
        requirements = f.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if constants.HYPHEN_E_DOT in requirements:
            requirements.remove(constants.HYPHEN_E_DOT)

        return requirements

setup(
    name="Next Word Predictor",
    version="0.0.1",
    author="Prince",
    author_email="princekhatri1013@gmail.com",
    packages=find_packages(),
    install_requires= get_requirements(REQUIREMENTS_FILE_PATH)
)

