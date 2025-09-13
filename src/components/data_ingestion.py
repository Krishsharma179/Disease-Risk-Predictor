from src.exception import CustomException
import os
import sys
import pandas as pd
from   sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logging
from src.components.data_transformation import Init_DatatransFormation
from src.components.diabetes_model_training import initiate_model_trainer

logging.info("entered the class")
@dataclass
class DataingestionConfig:
        train_diabetes_data:str=os.path.join('artifact','train_diabetes_data')
        test_diabetes_data:str=os.path.join('artifact','test_diabetes_data')
        raw_diabetes_data:str=os.path.join('artifact','raw_diabetes_data')
        
class Dataingestion:
   def __init__(self):
       self.data_ingestion_config=DataingestionConfig()
   def init_data_ingestion(self):
        """
            This function will return the path of train and test dataset it is the csv file 
        """
        try:
            df=pd.read_csv(r"C:\Users\krish sharma\OneDrive\Documents\diabetes_dataset2.csv")
        
            os.makedirs(os.path.dirname(os.path.join('artifact','train_diabetes_data')),exist_ok=True)
            logging.info("artifact dir made")
             
            df.to_csv(self.data_ingestion_config.raw_diabetes_data,index=False)
            logging.info("Raw folder saved in the artifact folder")

            logging.info("Train_test_split startes")


            train_data,test_data=train_test_split(df,random_state=42,test_size=0.2)

            train_data.to_csv(self.data_ingestion_config.train_diabetes_data,index=False,header=True)
            logging.info("train folder saved in the artifact folder")

            test_data.to_csv(self.data_ingestion_config.test_diabetes_data,index=False,header=True)
            logging.info("test folder saved in the artifact folder")
         

            return(
                self.data_ingestion_config.train_diabetes_data,
                self.data_ingestion_config.test_diabetes_data
            )

        except Exception as e:
          raise CustomException(e)

obj=Dataingestion()
train_data_path,test_data_path=obj.init_data_ingestion()
data_transformation=Init_DatatransFormation()
train_arr,test_arr,diabetes_processor_file_path=data_transformation.initiate_data_transformation(train_data_path,test_data_path)
model_trainer=initiate_model_trainer()
print(model_trainer.initiate_model_training(train_arr,test_arr))


