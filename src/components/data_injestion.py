# Data ingestion is to read the data 
'''
The expected output is the train and test data split
'''

import os
import sys
from typing import Any
from src.exception import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation

## Initialize the data ingestion configuration

@dataclass
class DataIngestionConfig(object):
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    raw_data_path:str = os.path.join('artifacts', 'raw.csv')

## create a class for data ingestion

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def intiate_data_ingestion(self):
        logging.info('Data ingestion method starts')
        try:
            df = pd.read_csv(os.path.join('notebooks/data', 'gemstone.csv'))
            logging.info('Data set read as pandas dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index = False)
            
            logging.info("Do the Train test split")
            train_set, test_set = train_test_split (df, test_size=0.30)

            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info("Data ingestion of data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Exception occured at data ingestion stage")
            raise CustomException(e, sys)
        

## Run data ingestion

if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path, test_data_path = obj.intiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
