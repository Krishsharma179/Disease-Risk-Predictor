from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import os
from src.components.cad_data_transformation import Init_DatatransFormation
from src.components.cad_model_training import initiate_model_trainer
logging.info('importion done')
@dataclass
class Datafilepath:
        train_cad_data:str=os.path.join('artifact','train_cad_data')
        test_cad_data:str=os.path.join('artifact','test_cad_data')
        raw_cad_data:str=os.path.join('artifact','raw_cad_data')
class Dateingestion:
        def __init__(self):
                self.filepath=Datafilepath()
        def initiate_data_ingestion(self):
                try:
                    df=pd.read_csv(r'C:\Users\krish sharma\OneDrive\Documents\cad_dataset')
                    logging.info('read the dataframe')
                    os.makedirs(os.path.dirname(os.path.join('artifact','train_cad_data')),exist_ok=True)
                    logging.info('made the directory')

                    df.to_csv(self.filepath.raw_cad_data,index=False)
                    logging.info('raw data folder created')
                    
                    logging.info('train test slit started ')

                    train_cad_data,test_cad_data=train_test_split(df,random_state=42,test_size=0.27)
                    logging.info("data ingestion of tain and test started")
                    train_cad_data.to_csv(self.filepath.train_cad_data,index=False)
                    test_cad_data.to_csv(self.filepath.test_cad_data,index=False)

                    return(self.filepath.train_cad_data,
                           self.filepath.test_cad_data)
                except Exception as e:
                       raise CustomException(e)    

data_ingestion=Dateingestion()
train_cad_data,test_cad_data=data_ingestion.initiate_data_ingestion()
data_transformation=Init_DatatransFormation()
train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_cad_data,test_cad_data)
# print(test_arr)
model_trainer=initiate_model_trainer()
print(model_trainer.initiate_model_training(train_arr,test_arr))

                      