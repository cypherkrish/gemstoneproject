import os
import sys

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging

import pickle


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.ino("Exception occured while generating pkl file")
        raise CustomException(e, sys)
    

def evaluate_model (X_train, y_train, X_test, y_test, models):

    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]

            # Train the model
            model.fit(X_train, y_train)


            # Predict the model for training data
            y_train_predict = model.predict(X_train)

            # Predict the model fo testing data
            y_test_predict = model.predict(X_test)

            # Determine the R2 score

            train_model_score = r2_score(y_train, y_train_predict)
            test_model_score = r2_score(y_test, y_test_predict)

            report[list(models.keys())[i]] = test_model_score

        return report


    except Exception as e:
        logging.ino("Exception occured ie evaluate model function")
        raise CustomException(e, sys)

    '''
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_squre = r2_score(true, predicted)
    return mae, rmse, r2_squre
    '''

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        logging.info("Exception occured in the load_object uder utils")
        raise CustomException(e, sys)