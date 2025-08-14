import os
import sys
import dill
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def save_object(file_path:str,obj):
    """
    This function save the object
    """
    try:
        file_dir=os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)
        with open(file_path,"wb") as file_object:
            dill.dump(obj,file_object) 
    except Exception as e:
        raise CustomException(e)

def evaluate_model(x_train,y_train,x_test,y_test,models:dict,param):
     score={}
     try:
          for i in range(len(list(models))):
            model=list(models.values())[i]
            model.fit(x_train,y_train)

            para=param[list(param.keys())[i]]
            

            gd=GridSearchCV(model,para,cv=3)
            gd.fit(x_train,y_train)

            model.set_params(**gd.best_params_)
            model.fit(x_train,y_train)

            y_pred=model.predict(x_test)
            r2score=r2_score(y_test,y_pred)

            score[list(models.keys())[i]]=r2score

          return score
     except Exception as e :
         raise CustomException(e)