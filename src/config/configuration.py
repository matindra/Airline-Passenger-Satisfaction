import os
from src.constant.constant import *
from src.exception import CustomException
from src.logger import logging


ROOT_DIR = ROOT_DIR_KEY
DATASET_PATH = os.path.join(ROOT_DIR,DATA_DIR, DATA_DIR_KEY)


# Data Ingestion related path variables

RAW_FILE_PATH = os.path.join(ROOT_DIR, ARTIFACT_DIR_KEY, DATA_INGESTION_KEY, 
                             CURRENT_TIME_STAMP, DATA_INGESTION_RAW_DATA_DIR_KEY, RAW_DATA_DIR_KEY)


TRAIN_FILE_PATH = os.path.join(ROOT_DIR, ARTIFACT_DIR_KEY, DATA_INGESTION_KEY,
                           CURRENT_TIME_STAMP,DATA_INGESTION_INGESTED_DIR_NAME_KEY,TRAIN_DATA_DIR_KEY)

TEST_FILE_PATH = os.path.join(ROOT_DIR, ARTIFACT_DIR_KEY, DATA_INGESTION_KEY,
                           CURRENT_TIME_STAMP,DATA_INGESTION_INGESTED_DIR_NAME_KEY,TEST_DATA_DIR_KEY)