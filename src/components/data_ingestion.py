import os, sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException

from src.constant.constants import *
from src.config.configuration import *

from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig():
    raw_data_path:str = RAW_FILE_PATH
    train_data_path:str = TRAIN_FILE_PATH
    test_data_path:str = TEST_FILE_PATH


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    
    def inititate_data_ingestion(self):
        logging.info(f"{'>>'*10} Data Ingestion Started {'<<'*10}")
        


        try:
            logging.info(f"Reading raw data from dataset path : {DATASET_PATH}")
            df = pd.read_csv(DATASET_PATH)
            logging.info("Raw Data Reading completed")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)


            logging.info("Initiating train test split ")
            train_set, test_set = train_test_split(df, test_size = .30, random_state=50)
            logging.info("Data splitted into train and test")


            logging.info(f"Creating train data at path : {TRAIN_FILE_PATH}")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok =True)
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            logging.info(f"train data created at path : {TRAIN_FILE_PATH}")


            logging.info(f"Creating test data at path : {TRAIN_FILE_PATH}")
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path),exist_ok =True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            logging.info(f"test data created at path : {TRAIN_FILE_PATH}")

            logging.info(f"{'>>'*10} Data Ingestion Completed{'<<'*10}")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            logging.info("Error occured in data ingestion stage")
            raise CustomException(e, sys)


if __name__ =="__main__":

    obj = DataIngestion()
    train_data_path , test_data_path = obj.inititate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path , test_data_path)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initate_model_training(train_arr, test_arr))


# src/components/data_ingestion.py
