import sys
import os

import numpy as np
import pandas as pd


from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
from dataclasses import dataclass 

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig(object):
    trained_model_file_path =  os.path.join("artifacts", "model.pkl")

class ModelTrainer(object):
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training (self, train_arr, test_arr):
        try:
            logging.info("Splitting dependent and independent fetures from train and test arrays")

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            ## Train multiple models 
            models = {
                'LinearRegression':LinearRegression(),
                'LassoRegression':Lasso(),
                'RidgeRegression':Ridge(),
                'ElasticNet':ElasticNet()
            }

            model_report:dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            print("Model report")
            print("\n================================================================================================")
            print(model_report)
            print("\n================================================================================================")
            logging.info(f'model_report: {model_report}')

            # Get the best model out of the given models
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_mode = models[best_model_name]

            print(f"Best mode found, Best model name:{best_mode}, with R2 score: {best_model_score}")
            print("\n================================================================================")

            logging.info(f"Best mode found, Best model name: {best_model_score}, with R2 score: {best_model_score}")
            
            # Generate the plk fine for model trainer

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_mode
            )

        except Exception as e:
            logging.info("Error occured in the initiate_model_training")
            raise CustomException(e, sys)