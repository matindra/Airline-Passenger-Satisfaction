# Basic Import
import os, sys
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.config.configuration import MODEL_FILE_PATH
from src.utils import save_object
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, classification_report,precision_score, recall_score, f1_score, roc_auc_score,roc_curve 
from dataclasses import dataclass



@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = MODEL_FILE_PATH





class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def evaluate_model(self,X_train,y_train,X_test,y_test,models,params):
        try:
            report = {}
            
            for i in range(len(models)):
                model = list(models.values())[i]
                para = params[list(models.keys())[i]]

                grid = GridSearchCV(model, para, cv = 3, n_jobs=-1,verbose=2)
                grid.fit(X_train, y_train)
                
                model.set_params(**grid.best_params_)
                model.fit(X_train,y_train)
                
                y_test_pred = model.predict(X_test)
                
                test_model_score = accuracy_score(y_test,y_test_pred)
                
                
                report[list(models.values())[i]] = test_model_score
            
            return report
                
        except Exception as e:
            logging.info("Exception occure while evaluation of model")
            raise CustomException(e,sys)
        




    def initate_model_training(self,train_array,test_array):
        try:
            logging.info(f"{'>>'*10} Model Training started {'<<'*10}")
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1])
            

            models={

            'Logistic':LogisticRegression(),          
            # 'DecisionTree':DecisionTreeClassifier(),
            # 'Gradient Boosting':GradientBoostingClassifier(),
            # 'Random Forest':RandomForestClassifier(),
            # 'XGB Classifier':XGBClassifier(),
            'KNN neighbour':KNeighborsClassifier()
            }


            params = {

                "Logistic":{
                    "class_weight":["balanced"],
                    'penalty': ['l1', 'l2'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'solver': ['liblinear', 'saga']
                },

                # "DecisionTree":{
                #     "class_weight":["balanced"],
                #     "criterion":['gini',"entropy","log_loss"],
                #     "splitter":['best','random'],
                #     "max_depth":[3,4,5,6],
                #     "min_samples_split":[2,3,4,5],
                #     "min_samples_leaf":[1,2,3],
                #     "max_features":["auto","sqrt","log2"]
                # },


                # "Gradient Boosting":{
                #     "learning_rate":[ 0.1, 0.05],
                #     "n_estimators":[50,100],
                #     "max_depth":[10, 8 ]

                # },



                # "Random Forest":{
                    
                #     'n_estimators': [20,  30,50,100],
                #     'max_depth': [10, 8, 5,None],
                #     'min_samples_split': [2, 5, 10],
                #     'criterion':["gini"]


                # },

                # "XGB Classifier":{
                #     'max_depth': [ 5, 7],
                #     'learning_rate': [0.1, 0.01],
                #     'n_estimators': [100, 200],
            
                #     'reg_alpha': [ 0.1, 0.5],
                #     'reg_lambda': [ 1, 10]
                # },


                "KNN neighbour":{
                        'n_neighbors': [3, 5, 7],
                        'weights': ['uniform', 'distance'],
                        
                    }
                
            }
            
           
            
            model_report:dict=self.evaluate_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                                            models=models,params=params)
          
            

            print(f"model_report: {model_report}")

            df = pd.DataFrame(list(model_report.items()), columns=['Model', 'Accuracy'])
            logging.info(f" model report {pd.DataFrame(list(model_report.items()), columns=['Model', 'Accuracy']) }")
            print(df)
            
            print('\n======================================================================\n')

            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            logging.info(f"best model score: {best_model_score}")

            best_model_name = list(models.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            # logging.info(f"{plot_confusion_matrix(best_model_name, X_test, self.y_test_pred, cmap='Blues', values_format='d')}")

            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model

            )

            logging.info(f"{'>>'*10} Model Training Completed {'<<'*10}")
          

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)