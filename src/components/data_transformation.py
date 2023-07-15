# Data transformation step
'''
Usuallu this step is to do 
1. Feature engineering
2. Handeling 
    Missing values, 
    outliers, 
    feature scaling,
    categrical, numerical fetures

Expected the Data transformation along with the pickle file generation.
'''
import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from src.exception import logging
from src.exception import CustomException

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            
            logging.info("Data Transformation step initiated")
            
            # seperte the categorical and numerical variables
            categorical_columns = ['cut', 'color', 'clarity']
            numerical_columns = ['carat', 'depth', 'table', 'x', 'y', 'z']

            # Variaous uniqueue categorical vlues for each categorical feature. 

            cut_category = ['Premium', 'Very Good', 'Ideal', 'Good', 'Fair']
            color_category = ['F', 'J', 'G', 'E', 'D', 'H', 'I']
            clarity_category = ['VS2', 'SI2', 'VS1', 'SI1', 'IF', 'VVS2', 'VVS1', 'I1']

            logging.info("Pipeling initiated")
            

            # Numerical pipeline

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scalar', StandardScaler())
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('impute', SimpleImputer(strategy='most_frequent')),
                    ('OrdinalEncoder', OrdinalEncoder(categories=[cut_category, color_category, clarity_category])),
                    ('scalar', StandardScaler())

                ]

            )

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)

                ]
            )

            logging.info("Pipeline completed")

            return preprocessor

        except Exception as e:
            logging.info("Error in data transformation section")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info(f'Train Dataframe Head: \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe HEad: \n{test_df.head().to_string()}')

            logging.info("Gettignt the prepreocessing object")

            preprocessing_obj = self.get_data_transformation_obj()

            
            ## Seperate the input and targe features

            target_column_name = 'price'
            drop_columns = [target_column_name, 'id']

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]


            ## Transforming the data using the preprocessing object
            logging.info("Applying the preporcessing object on the train and test datasets")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)


            # Comcatinatig using the numpy array
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Call utlis function - save_object to save pkl file

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessor pickle file got saved")

            return (
                train_arr, 
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Exception occured in initiate data transformation setp")
            raise CustomException(e, sys)
        