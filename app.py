from flask import Flask,render_template,request
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.pipeline.diabetes_predict_pipeline import Customdata
from src.pipeline.diabetes_predict_pipeline import Predictpipeline


application=Flask(__name__)
app=application

@app.route('/')
def index():
   return render_template('home.html')

@app.route("/diabetes_predict",methods=["POST","GET"])
def prediction():
   if request.method=="GET":
      return render_template('diabetes.html')
   else:
      data=Customdata(
         Gender=request.form.get('Gender'),
         AGE=request.form.get('AGE'),
         Urea=request.form.get('Urea'),
         Cr=request.form.get('CR'),
         Chol=request.form.get('Chol'),
         TG=request.form.get('TG'),
         HDL=request.form.get('HDL'),
         LDL=request.form.get('LDL'),
         VLDL=request.form.get('VLDL'),
         BMI=request.form.get('BMI')

      )
      pred_data=data.covert_data_into_df()

      predict_data=Predictpipeline()
      result,result_prob=predict_data.predict(pred_data)

      result_lst=result_prob[0]

      def largest_element(lst):
         largest=lst[0]
         place=0
         for i in range(0,3):
            if lst[i]>largest:
                  largest=lst[i]
                  place+=i

         return largest   
      
      largest=largest_element(result_lst)

      def place(lst,element):
         names={
             0:"non-diabetic",
             1:"diabetic",
             2:"prediabetic"

         }
         for i in range(len(lst)):
            if lst[i]==element:
                  return names[i]       

 

      return render_template('diabetes.html',val1=largest_element(result_lst)*100,val2=place(result_lst,largest))
   



if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)   