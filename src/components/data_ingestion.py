import os, sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException

from src.constant.constant import *
from src.config.configuration import *

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig():
    raw_data_path:str = RAW_FILE_PATH
    train_data_path:str = TRAIN_FILE_PATH
    test_data_path:str = TEST_FILE_PATH


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    
    def inititate_data_ingestion(self):
        logging.info("Data Ingestion started")
        


        try:
            logging.info(f" Reading data from dataset path : {DATASET_PATH}")
            df = pd.read_csv(DATASET_PATH)
            logging.info("Data Reading completed")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Data splitted into train and test")

            train_set, test_set = train_test_split(df, test_size = .30, random_state=50)


            logging.info(f"Creating train data at path : {TRAIN_FILE_PATH}")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok =True)
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            logging.info(f"train data created at path : {TRAIN_FILE_PATH}")


            logging.info(f"Creating test data at path : {TRAIN_FILE_PATH}")
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path),exist_ok =True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            logging.info(f"test data created at path : {TRAIN_FILE_PATH}")

            logging.info("Data Ingestion completed")

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


# src/components/data_ingestion.py
