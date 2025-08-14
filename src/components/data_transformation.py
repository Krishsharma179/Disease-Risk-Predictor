import sys
import os
from src.logger import logging
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute  import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utlis import save_object
from dataclasses import dataclass
logging.info("importing done")

@dataclass
class DatatransFromationConfig:
    transformdata=os.path.join("artifact","diabetes_preprocessor.pkl")

class Init_DatatransFormation:
    def __init__(self):
        self.data_transformation_config=DatatransFromationConfig()

    logging.info("Started makeing pipeline")
    def data_transformaton(self):
        """
        This function is responsible for the Data Transformation 
        """
        cat_features=['Gender']
        num_feature=['AGE','Urea','Cr','Chol','TG','HDL',"LDL",'VLDL','BMI']

        cat_pipeline=Pipeline(
            steps=[("SimpleImputer",SimpleImputer(strategy="most_frequent")),
                   ("OneHotEncoding",OneHotEncoder())]
        )

        num_pipeline=Pipeline(
            steps=[
                ("SimpleIMputer",SimpleImputer(strategy="median")),
                ("Standardization",StandardScaler())
            ]
        )

        column_transformer=ColumnTransformer(
            [
                ("cat_transformation",cat_pipeline,cat_features),
                 ("num_transformation",num_pipeline,num_feature)
            ]
        )

        return column_transformer

    def initiate_data_transformation(self,train_data,test_data):
            preprocessor_obj=self.data_transformaton()
            try:
                train_data=pd.read_csv(train_data)
                test_data=pd.read_csv(test_data)

                target='CLASS'
                
                train_independent_df=train_data.drop(target,axis=1)
                train_dependent_df=train_data[target]
            
                test_independent_df=test_data.drop(target,axis=1)
                test_dependent_df=test_data[target]
                logging.info("Starting data transformation")

                train_transformed_independent_df=preprocessor_obj.fit_transform(train_independent_df)
                test_transformed_independent_df=preprocessor_obj.transform(test_independent_df)

                train_arr=np.c_[train_transformed_independent_df,np.array(train_dependent_df)]
                test_arr=np.c_[test_transformed_independent_df,np.array(test_dependent_df)]

                logging.info("completed data transformation")

                save_object(
                     file_path=self.data_transformation_config.transformdata,
                     obj=preprocessor_obj

                )
                return(train_arr,test_arr,self.data_transformation_config.transformdata)
                


                


            except Exception as e:
               raise  CustomException(e)
                
            

            


        

