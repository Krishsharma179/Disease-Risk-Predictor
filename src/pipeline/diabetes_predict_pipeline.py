import os
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from src.utlis import load_object
from dataclasses import dataclass

@dataclass
class Predictpipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_file_path="artifact\diabetes_model.pkl"
            preprocessor_file_path="artifact\diabetes_preprocessor.pkl"
            model=load_object(file_path=model_file_path)
            preprocessor=load_object(file_path=preprocessor_file_path)

            scaled_data=preprocessor.transform(features)
            pred=model.predict(scaled_data)
            pred_probs = model.predict_proba(scaled_data)

            return pred,pred_probs
        except Exception as e:
            raise CustomException(e)
        
class Customdata:
    def __init__(self,Gender:str,AGE:int,Urea:int,Cr:int,Chol:int,TG:int,HDL:int,LDL:int,VLDL:int,BMI:int):
        self.Gender=Gender
        self.AGE=AGE
        self.Urea=Urea        
        self.Cr=Cr        
        self.Chol=Chol        
        self.TG=TG        
        self.HDL=HDL        
        self.LDL=LDL        
        self.VLDL=VLDL        
        self.BMI=BMI

    def covert_data_into_df(self):
        dataframe={
            "Gender":[self.Gender],
            "AGE":[self.AGE],
            "Urea":[self.Urea],
            "Cr":[self.Cr],
            "Chol":[self.Chol],
            "TG":[self.TG],
            "HDL":[self.HDL],
            "LDL":[self.LDL],
            "VLDL":[self.VLDL],
            "BMI":[self.BMI]
        }

        df=pd.DataFrame(dataframe)

        return df            
        