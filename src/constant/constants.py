import os
from datetime import datetime


def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"


CURRENT_TIME_STAMP = get_current_time_stamp()

ROOT_DIR_KEY = os.getcwd()

DATA_DIR = "data"
DATASET_KEY = "airline_dataset.csv"


# Artifact related constants

ARTIFACT_DIR_KEY = "Artifact"

# Data Ingestion related constants

DATA_INGESTION_KEY = 'data_ingestion'
DATA_INGESTION_RAW_DATA_DIR_KEY= 'raw_data_dir'
DATA_INGESTION_INGESTED_DIR_NAME_KEY= 'ingested_dir'
RAW_DATA_DIR_KEY = 'raw.csv'
TRAIN_DATA_DIR_KEY = 'train.csv'
TEST_DATA_DIR_KEY = 'test.csv' 


# Data Transformation related constants

DATA_TRANSFORMATION_ARTIFACT = 'data_transformation'
DATA_PREPROCESSED_DIR='preprocessed'
DATA_TRANSFORMATION_PREPROCESSING_OBJ = 'preprocessor.pkl'
DATA_TRANSFORMED_DIR = 'transformed_data'
TRANSFORMED_TRAIN_DIR_KEY = 'train.csv'
TRANSFORMED_TEST_DIR_KEY = 'test.csv'
THRESHOLD_KEY = 0.85
