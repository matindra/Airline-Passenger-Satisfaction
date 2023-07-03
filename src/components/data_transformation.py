import os,sys
from src.exception import CustomException
from src.logger import logging
from src.constant.constants import *
from src.config.configuration import *   #PREPROCESSING_OBJ_PATH,TRANSFORMED_TRAIN_FILE_PATH,TRANSFORMED_TEST_FILE_PATH
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


    def remove_outliers_IQR(self, col, df):
        try:
            logging.info('outlier handling code')
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            iqr = Q3 - Q1
            upper_limit = Q3 + 1.5 * iqr
            lower_limit = Q1 - 1.5 * iqr
            df.loc[(df[col]>upper_limit), col]= upper_limit
            df.loc[(df[col]<lower_limit), col]= lower_limit
            return df
        
        except Exception as e:
            raise CustomException(e, sys) from e 
        


    def get_data_transformation_object(self):
        try:
            logging.info("Initiating data transformation")


            
            # Separating Numerical features
            numerical_columns = ['Age', 'Flight_Distance', 'Inflight_wifi_service',
       'Departure_Arrival_time_convenient', 'Ease_of_Online_booking',
       'Gate_location', 'Food_and_drink', 'Online_boarding', 'Seat_comfort',
       'Inflight_entertainment', 'On_board_service', 'Leg_room_service',
       'Baggage_handling', 'Checkin_service', 'Inflight_service',
       'Cleanliness', 'Departure_Delay_in_Minutes',
       'Arrival_Delay_in_Minutes']

            # Separating categorical features
            categorical_columns = ['Gender', 'Customer_Type', 'Type_of_Travel', 'Class', 'satisfaction']

            
            numerical_pipeline=Pipeline(steps=[
                ('impute',SimpleImputer()),
                ('scaler',StandardScaler()),
                ('transformer', PowerTransformer(method='yeo-johnson', standardize=False))
            ])

            # categorical_pipeline=Pipeline(steps=[
            #     ('impute',SimpleImputer(strategy='most_frequent')),
            #     ('onehot',OneHotEncoder(handle_unknown='ignore')),
            #     ('scaler',StandardScaler(with_mean=False))
            #     ])

            
            onehot_columns = ['Gender', 'Customer_Type', 'Type_of_Travel']
            ordinal_columns = ['Class']
            label_encoder_column = ['satisfaction']

            onehot_pipeline= Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('oridnal_encoder', OrdinalEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])


            ordinal_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ]
            )



            preprocessor =ColumnTransformer([
                ('numerical_pipeline',numerical_pipeline,numerical_columns),
                # ('category_pipeline',categorical_pipeline,categorical_columns)
                ('onehot_pipeline', onehot_pipeline, onehot_columns),
                ('ordinal_pipeline', ordinal_pipeline, ordinal_columns),

            ])

            return preprocessor
        
            logging.info('pipeline completed')

        except Exception as e:
            logging.info("Error getting data transformation object")
            raise CustomException(e,sys)
        


    # def correlation(self, dataset, threshold):
    #     try:
    #         logging.info(f"Initiating removal of highly_correlted_features")
    #         col_corr = set()  # Set of all the names of correlated columns
    #         corr_matrix = dataset.corr()
    #         for i in range(len(corr_matrix.columns)):
    #             for j in range(i):
    #                 if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
    #                     colname = corr_matrix.columns[i]  # getting the name of column
    #                     col_corr.add(colname)
                    
    #         return dataset.drop(col_corr,axis=1)
        
    #     except Exception as e:
    #         raise CustomException(e, sys) from e 

        

    
    def initiate_data_transformation(self,train_path,test_path):
        try:

            logging.info(f"{'>>'*10} Data Transformation started {'<<'*10}")

            # Reading train and test file
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Reading train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')


            logging.info(f"columns in dataframe are: {train_df.columns}")

            logging.info(f"columns in dataframe are: {train_df.dtypes}")

            # logging.info(f"Initiating removal of highly_correlted_features in train")
            # train_df = self.correlation(dataset=train_df, threshold=THRESHOLD_KEY)

            # logging.info(f"Initiating removal of highly_correlted_features in test")
            # test_df = self.correlation(dataset=test_df, threshold=THRESHOLD_KEY)




            # Outlier removal from train and test

            # numerical_features = DATASET_KEY.select_dtypes(exclude="object").columns     # --->> This line is giving error

            numerical_features = ['Age', 'Flight_Distance', 'Inflight_wifi_service',
       'Departure_Arrival_time_convenient', 'Ease_of_Online_booking',
       'Gate_location', 'Food_and_drink', 'Online_boarding', 'Seat_comfort',
       'Inflight_entertainment', 'On_board_service', 'Leg_room_service',
       'Baggage_handling', 'Checkin_service', 'Inflight_service',
       'Cleanliness', 'Departure_Delay_in_Minutes',
       'Arrival_Delay_in_Minutes']
            


            for col in numerical_features:
                self.remove_outliers_IQR(col = col, df = train_df)
            
            logging.info('outlier capped on our train data')
            
            for col in numerical_features:
                self.remove_outliers_IQR(col = col, df = test_df)
            
            logging.info('outlier capped on our test data')


            preprocessing_obj = self.get_data_transformation_object()


            target_column_name = 'satisfaction'


            logging.info("Splitting train data into dependent X_Train and independent features y_train")
            # X_train = train_df.drop(['satisfaction'],axis=1)
            # y_train = train_df['satisfaction']
            X_train = train_df.drop(target_column_name, axis=1)
            y_train = train_df[target_column_name]


            logging.info("Splitting test data into dependent X_test and independent features y_test")
            X_test = test_df.drop(target_column_name,axis=1)
            y_test = test_df[target_column_name]


            logging.info(f"shape of X_Train {X_train.shape} and y_train {y_train.shape}")
            logging.info(f"shape of X_test {X_test.shape} and y_test {y_test.shape}")

            # Transforming using preprocessor obj
            
            X_train = preprocessing_obj.fit_transform(X_train)
            X_test = preprocessing_obj.transform(X_test)


            logging.info("Applying preprocessing object on training and testing datasets.")            


            train_arr =np.c_[X_train,np.array(y_train)]
            test_arr =np.c_[X_test,np.array(y_test)]


            logging.info("converting train_arr and test_arr to dataframe")
            df_train= pd.DataFrame(train_arr)
            df_test = pd.DataFrame(test_arr)

            os.makedirs(os.path.dirname(self.transformation_config.transformed_train_path),exist_ok=True)
            df_train.to_csv(self.transformation_config.transformed_train_path,index=False,header=True)

            logging.info("transformed_train_path")
            logging.info(f"transformed dataset columns : {df_train.columns}")

            os.makedirs(os.path.dirname(self.transformation_config.transformed_test_path),exist_ok=True)
            df_test.to_csv(self.transformation_config.transformed_test_path,index=False,header=True)

            logging.info("transformed_test_path")

            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj)
            
            logging.info("Preprocessor file saved")
            logging.info(f"{'>>'*10} Data Transformation Completed {'<<'*10}")
            
            return(train_arr,
                   test_arr,
                   self.transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys) from e 