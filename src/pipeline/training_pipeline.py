import os
import sys
import pandas as pd
import numpy as np

from src.logger import logging

from src.components.data_injestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer



## Run the total End to End project

## Step 1: Run Data Ingestion
## Step 2: Run Data transformation and genereate the preprocessor pkl file
## Step 3: Run Model trainer and generate the mode trainer pkl file.

if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path, test_data_path = obj.intiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_arr, test_arr)