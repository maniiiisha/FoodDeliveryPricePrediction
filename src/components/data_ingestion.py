import pandas as pd
import numpy as np
import os, sys
from src.logging import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from src.components.data_transformation import datatransformation, datatransformationconfig
from src.components.model_trainer import modeltrainer, modeltrainingconfig
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_path:str = os.path.join('artifacts','train.csv')
    test_path:str = os.path.join('artifacts','test.csv')
    raw_path:str = os.path.join('artifacts','raw.csv')

class dataingestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('data ingestion started')
        try:
            df = pd.read_csv(os.path.join('notebooks/data','finalTrain.csv'))
            data = df.drop(columns = 'ID')
            logging.info('Dataset read as pandas dataframe.')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_path), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_path, index=False)

            logging.info('train test split initiated')
            train_set, test_set = train_test_split(data, test_size=0.20, random_state=42)
            train_set.to_csv(self.ingestion_config.train_path, index=False)
            test_set.to_csv(self.ingestion_config.test_path, index=False)

            logging.info('data ingestion ended')

            return(self.ingestion_config.train_path, self.ingestion_config.test_path)
        
        except Exception as e:
            logging.info('error in data ingestion stage')
            raise CustomException(e,sys)

'''
if __name__ =='__main__':
    obj = dataingestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_trans = datatransformation()
    train_arr, test_arr, _ = data_trans.initiate_data_transformation(train_data, test_data)

    model_train = modeltrainer()
    model_train.initiate_model_training(train_arr, test_arr)
'''