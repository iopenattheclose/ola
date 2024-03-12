import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from dataclasses import dataclass

from src.components.data_preprocess_feature_eng import DataPreProcessingFE,DataPreProcessingFEConfig

from src.components.data_transformation import DataTransformation,DataTransformationConfig

from src.components.model_trainer import ModelTrainer,ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    #declaring path variable ie created in artifacts folder
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        # ingestion_config variable will consist of the this value defined in DataIngestionConfig class
        self.ingestion_config=DataIngestionConfig()


    def initiate_data_ingestion(self):
            logging.info("Entered the data ingestion method or component")
            try:
                #instead of csv file, this data can be fetched from any data source (viz mongodb)
                df=pd.read_csv('notebook/data/ola.csv')
                logging.info('Read the dataset as dataframe')

                #creating artifact folder with path as parameter
                os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

                #raw_csv_ie_complete_data_as_a_csv
                df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
                logging.info("Raw data is stored inside artifact folder")

            except Exception as e:
                raise CustomException(e,sys)
            
if __name__=="__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()

    data_PPFE = DataPreProcessingFE()
    preprocessed_df,train_data_path,test_data_path = data_PPFE.initiate_data_preprocessing()

    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(preprocessed_df,train_data_path,test_data_path)

    modeltrainer=ModelTrainer()
    print('Accuracy of Ensemble model on test data set: {:.5f}'.format(modeltrainer.initiate_model_trainer(train_arr,test_arr)))
    
