import os
import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredectPipeline(object):
    def __init__(self) -> None:
        pass

    def predict(self,features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            preprocessor =  load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)

            return pred
                                      

        except Exception as e:
            logging.info("Error occuredin the redict function under the predection pipeline")
            raise CustomException(e, sys)
        

class CustomData:
    def __init__(self,
                 carat:float,
                 cut:str,
                 color: str,
                 clarity: str,
                 depth: float,
                 table: float,
                 x: float,
                 y: float,
                 z: float):
        self.carat = carat
        self.cut = cut
        self.color = color
        self.clarity = clarity
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'carat': [self.carat], 
                'cut': [self.cut],
                'color': [self.color],
                'clarity':[self.clarity],
                'depth': [self.depth],
                'table': [self.table],
                'x': [self.x],
                'y': [self.y],
                'z': [self.z]
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Input dataframe is created')
            return df
        
        except Exception as e:
            logging.info("Exception in get_data_as_dataframe in predection pipeline")
            raise CustomException(e, sys)
        
        