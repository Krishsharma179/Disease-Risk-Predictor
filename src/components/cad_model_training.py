import os
import sys
import pickle as pkl

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier 
from xgboost import XGBRFClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# import warnings
# warnings.filterwarnings('ignore')




from sklearn.metrics import r2_score

from src.exception import CustomException

from src.logger import logging
from dataclasses import dataclass

from src.utlis import save_object
from src.utlis import evaluate_model


logging.info("All the file got imported")

@dataclass
class ModeltrainerConfig:
        trained_model_file_path=os.path.join("artifact","cad_model.pkl")

class initiate_model_trainer:
        logging.info("initaited model training")
        try:
                def __init__(self):
                        self.ModelTrainerConfig=ModeltrainerConfig()
                def  initiate_model_training(self,train_arr,test_arr):
                        x_train,x_test,y_train,y_test=(
                            train_arr[:,:-1],
                            test_arr[:,:-1],
                            train_arr[:,-1], 
                            test_arr[:,-1]
                               
                        )
                        models = {
                            "LogisticRegression": LogisticRegression(),
                            "KNeighborsClassifier": KNeighborsClassifier(),
                            "AdaBoostClassifier": AdaBoostClassifier(),
                            "RandomForestClassifier": RandomForestClassifier(),
                            # "DecisionTreeClassifier": DecisionTreeClassifier(),
                            # "GradientBoostingClassifier": GradientBoostingClassifier(),
                            # "XGBRFClassifier": XGBRFClassifier()
                        }

                        params_classifier = {
                            "LogisticRegression": 
                                {
                                    'penalty': ['l2', 'none'],
                                    'solver': ['lbfgs', 'newton-cg', 'sag', 'saga'],
                                    'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                                    'max_iter': [100, 500, 1000, 2500]
                                },
                                

                            "KNeighborsClassifier": {
                                'n_neighbors': [3, 5, 7, 9],
                                'weights': ['uniform', 'distance'],
                                'metric': ['euclidean', 'manhattan', 'minkowski']
                            },

                            "AdaBoostClassifier": {
                                'n_estimators': [50, 100, 150, 200],
                                'learning_rate': [0.01, 0.1, 0.5, 1.0],
                                'algorithm': ['SAMME', 'SAMME.R'],
                                'estimator': [None, DecisionTreeClassifier(max_depth=1)]  # decision stump option
                            },

                            "RandomForestClassifier": {
                                'n_estimators': [50, 100, 200],
                                'max_depth': [None, 10, 20, 30],
                                'min_samples_split': [2, 5, 10],
                                'min_samples_leaf': [1, 2, 4],
                                'max_features': ['sqrt', 'log2', None],  # 'auto' can be deprecated
                                'bootstrap': [True, False]
                            },

                            "DecisionTreeClassifier": {
                                'criterion': ['gini', 'entropy'],
                                'max_depth': [None, 10, 20, 30],
                                'min_samples_split': [2, 5, 10],
                                # 'min_samples_leaf': [1, 2, 4],
                                # 'max_features': ['sqrt', 'log2', None],  # Removed 'auto' to avoid warnings
                                # 'splitter': ['best', 'random']
                            },

                            # "GradientBoostingClassifier": {
                            #     'n_estimators': [50, 100, 200],
                            #     'learning_rate': [0.01, 0.05, 0.1, 0.2],
                            #     'max_depth': [3, 5, 7],
                            #     # 'min_samples_split': [2, 5],
                            #     # 'min_samples_leaf': [1, 2],
                            #     # 'subsample': [0.6, 0.8, 1.0],
                            #     # 'max_features': ['sqrt', 'log2', None]  # Removed 'auto' for consistency
                            # },

                            # "XGBRFClassifier": {
                            #     'n_estimators': [50, 100, 200],
                            #     'max_depth': [3, 5, 7],
                            # #     'learning_rate': [0.01, 0.1],
                            # #     'subsample': [0.6, 0.8, 1.0],
                            # #     'colsample_bynode': [0.6, 0.8, 1.0],
                            # #     'min_child_weight': [1, 3, 5]
                            # }
                        }

                        model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,param=params_classifier)

                        best_score=max(sorted(model_report.values()))

                        if best_score<0.6:
                            raise CustomException("NO best model")
                        
                        # getting best model name
                        best_model_name=list(model_report.keys())[list(model_report.values()).index(best_score)]
                        # .index will give the index of best model score in model report
                        
                        best_model=models[best_model_name]

                        save_object(
                               file_path=self.ModelTrainerConfig.trained_model_file_path,
                               obj=best_model
                               
                        )
                          
                        best_model.fit(x_train,y_train)
                        y_pred=best_model.predict(x_test)

                        accuracy=accuracy_score(y_test,y_pred)

                        return (accuracy,best_model)

       
        except Exception as e:
                raise CustomException(e)
                                