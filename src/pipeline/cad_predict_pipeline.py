import os
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from src.utlis import load_object
from dataclasses import dataclass
# age,sex,chest_pain_type,resting_bp_s,cholesterol,fasting_blood_sugar,resting_ecg,max_heart_rate,exercise_angina,oldpeak,ST_slope,target
@dataclass
class Cad_predictpipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_file_path="artifact\cad_model.pkl"
            preprocessor_file_path="artifact\cad_preprocessor.pkl"
            model=load_object(file_path=model_file_path)
            preprocessor=load_object(file_path=preprocessor_file_path)

            scaled_data=preprocessor.transform(features)
            pred=model.predict(scaled_data)
            pred_probs = model.predict_proba(scaled_data)

            return pred,pred_probs
        except Exception as e:
            raise CustomException(e)
        
class Cad_customdata:
    def __init__(self,sex:int,age:int,chest_pain_type:int,resting_bp_s:int,cholesterol:int,fasting_blood_sugar:int,resting_ecg:int,max_heart_rate:int,exercise_angina:int,oldpeak:int,ST_slope:int):
        self.sex=sex
        self.age=age
        self.chest_pain_type=chest_pain_type
        self.resting_bp_s=resting_bp_s        
        self.cholesterol=cholesterol        
        self.fasting_blood_sugar=fasting_blood_sugar        
        self.resting_ecg=resting_ecg        
        self.max_heart_rate=max_heart_rate        
        self.exercise_angina=exercise_angina        
        self.oldpeak=oldpeak        
        self.ST_slope=ST_slope

    def covert_data_into_df(self):
        dataframe={
            "age":[self.age],
            "sex":[self.sex],
            "chest_pain_type":[self.chest_pain_type],
            "resting_bp_s":[self.resting_bp_s],
            "cholesterol":[self.cholesterol],
            "fasting_blood_sugar":[self.fasting_blood_sugar],
            "resting_ecg":[self.resting_ecg],
            "max_heart_rate":[self.max_heart_rate],
            "exercise_angina":[self.exercise_angina],
            "oldpeak":[self.oldpeak],
            "ST_slope":[self.ST_slope]
        }

        df=pd.DataFrame(dataframe)

        return df            
        