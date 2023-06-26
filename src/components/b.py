import os,sys
from src.exception import CustomException
from src.logger import logging
from src.constant.constants import *
from src.config.configuration import * 
from dataclasses import dataclass


import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from src.utils import save_object

@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path=PREPROCESSING_OBJ_PATH
    transformed_train_path = TRANSFORMED_TRAIN_FILE_PATH
    transformed_test_path = TRANSFORMED_TEST_FILE_PATH



class DataTransformation():
    def __init__(self):
        self.transformation_config=DataTransformationConfig()

        


    def get_data_transformation_object(self):
        try:
            logging.info("Initiating data transformation")
            
            
            
            # Separating Numerical features
            numerical_columns = ['Age', 'Flight Distance', 'Inflight wifi service','Departure/Arrival time convenient',
                                 'Ease of Online booking','Gate location', 'Food and drink', 'Online boarding',
                                 'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service', 
                                 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness',
                                 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
            

            # Separating categorical features
            categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']

            
            
            numerical_pipeline=Pipeline(steps=[
                ('impute',SimpleImputer()),
                ('scaler',StandardScaler()),
                ('transformer', PowerTransformer(method='yeo-johnson', standardize=False))
            ])

            categorical_pipeline=Pipeline(steps=[
                ('impute',SimpleImputer(strategy='most_frequent')),
                ('onehot',OneHotEncoder(handle_unknown='ignore')),
                ('scaler',StandardScaler(with_mean=False))
                ])

            preprocessor =ColumnTransformer([
                ('numerical_pipeline',numerical_pipeline,numerical_columns),
                ('category_pipeline',categorical_pipeline,categorical_columns)
            ])

            return preprocessor
        
            logging.info('pipeline completed')

        except Exception as e:
            logging.info("Error getting data transformation object")
            raise CustomException(e,sys)
        

    
    def initiate_data_transformation(self,train_path,test_path):
        try:


            # Reading train and test file
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)


            numerical_features = ['Age','Flight Distance','Inflight wifi service','Departure/Arrival time convenient','Ease of Online booking',
                    'Food and drink','Online boarding','Seat comfort','Inflight entertainment','On-board service','Leg room service','Baggage handling',
                    'Checkin service','Inflight service','Cleanliness']

            
            logging.info('outlier capped on our test data')


            preprocessing_obj = self.get_data_transformation_object()



            logging.info("Splitting train data into dependent and independent features")
            X_train = train_df.drop(['satisfaction'],axis=1)
            y_train = train_df['satisfaction']


            logging.info("Splitting test data into dependent and independent features")
            X_test = test_df.drop(['satisfaction'],axis=1)
            y_test = test_df['satisfaction']
            
            X_train = preprocessing_obj.fit_transform(X_train)
            X_test = preprocessing_obj.transform(X_test)


            logging.info("Applying preprocessing object on training and testing datasets.")            


            train_arr =np.c_[X_train,np.array(y_train)]
            test_arr =np.c_[X_test,np.array(y_test)]


            logging.info("converting train_arr and test_arr to dataframe")
            df_train= pd.DataFrame(train_arr)
            df_test = pd.DataFrame(test_arr)


            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj)
            
            logging.info("Preprocessor file saved")
            logging.info("Data transformation completed")
            
            return(train_arr,
                   test_arr,
                   self.transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys) from e 