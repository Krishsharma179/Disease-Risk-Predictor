from flask import Flask,render_template,request
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.pipeline.diabetes_predict_pipeline import Customdata
from src.pipeline.diabetes_predict_pipeline import Predictpipeline
from src.pipeline.cad_predict_pipeline import Cad_customdata
from src.pipeline.cad_predict_pipeline import Cad_predictpipeline

# age,sex,chest_pain_type,resting_bp_s,cholesterol,fasting_blood_sugar,resting_ecg,max_heart_rate,exercise_angina,oldpeak,ST_slope,target

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

@app.route('/predict_cad',methods=['GET','POST'])
def pred():
   if request.method=="GET":
      return render_template('cad.html')
   else:
      data=Cad_customdata(
         age=request.form.get('age'),
         sex=request.form.get('sex'),
         chest_pain_type=request.form.get('chest_pain_type'),
         resting_bp_s=request.form.get('resting_bp_s'),
         cholesterol=request.form.get('cholesterol'),
         fasting_blood_sugar=request.form.get('fasting_blood_sugar'),
         resting_ecg=request.form.get('resting_ecg'),
         max_heart_rate=request.form.get('max_heart_rate'),
         exercise_angina=request.form.get('exercise_angina'),
         oldpeak=request.form.get('oldpeak'),
         ST_slope=request.form.get('ST_slope')

      )
      pred_data=data.covert_data_into_df()

      predict_data=Cad_predictpipeline()
      result,result_prob=predict_data.predict(pred_data)

      result_pro=result_prob[0]
      name={
          0:"absence of disease",
          1:"Presence of this dieases"
      }



      def largest(lst):
         large=0
         index=0
         first_element=lst[0]
         second_element=lst[1]

         if first_element>second_element:
                  large=first_element
                  index=0
         else:
                  large=second_element
                  index=1
         return large ,index        

      r,index=largest(result_pro)
      result=round(r, 2)
      ind=name[index]       
                        
    
 

      return render_template('cad.html',result=result*100,index=ind)

if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)   