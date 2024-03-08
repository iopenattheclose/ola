import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,MaxAbsScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self,df):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            target_column_name='Target'
            df.drop(columns=[target_column_name],axis=1,inplace=True)
            num_features = df.select_dtypes(exclude="object").columns
            cat_features = df.select_dtypes(include="object").columns

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",MaxAbsScaler())
                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",MaxAbsScaler())
                ]
            )

            logging.info(f"Categorical columns: {cat_features}")
            logging.info(f"Numerical columns: {num_features}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,num_features),
                ("cat_pipelines",cat_pipeline,cat_features)
                ]
            )

            return preprocessor
        

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,df,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object(df)

            target_column_name='Target'

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]


            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            logging.info("input feature train arr preprocessing object")
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            logging.info("input feature test arr preprocessing object")


            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        

    