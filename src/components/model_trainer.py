import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.utils import class_weight
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix

from sklearn.preprocessing import OneHotEncoder,StandardScaler,MaxAbsScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")

            param = {'max_depth':[2,3,4,5,6,7], 'n_estimators':[50,100,150,200,250]}
            random_forest = RandomForestClassifier(class_weight ='balanced')

            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            random_forest = RandomForestClassifier(class_weight ='balanced')

            c = GridSearchCV(random_forest,param,cv=3,scoring='f1')

            c.fit(X_train,y_train)

            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = c
            )

            def display(results):
                print(f'Best parameters are : {results.best_params_}')
                print(f'The score is : {results.best_score_}')
        
            display(c)
            y_pred = c.predict(X_test)

            print(classification_report(y_test, y_pred))
            cm = confusion_matrix(y_test, y_pred)
            print(cm)

        except Exception as e:
            raise CustomException(e,sys)

    