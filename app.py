from flask import Flask,request,render_template
import numpy as np
import pandas as pd



from sklearn.preprocessing import StandardScaler
from src.pipelines.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
    Age=request.form.get('Age'),
    Gender=request.form.get('Gender'),
    Education=request.form.get('Education'),
    Income=request.form.get('Income'),
    Joining_Designation=request.form.get('Joining_Designation'),
    Grade=request.form.get('Grade'),
    Total_Business_Value=request.form.get('Total_Business_Value'),
    Last_Quarterly_Rating=request.form.get('Last_Quarterly_Rating'),
    Quarterly_Rating_Increased=request.form.get('Quarterly_Rating_Increased'),
    Income_Increased=request.form.get('Income_Increased'),
    City_C1=request.form.get('City_C1'),
    City_C10=request.form.get('City_C10'),
    City_C11=request.form.get('City_C11'),
    City_C12=request.form.get('City_C12'),
    City_C13=request.form.get('City_C13'),
    City_C14=request.form.get('City_C14'),
    City_C15=request.form.get('City_C15'),
    City_C16=request.form.get('City_C16'),
    City_C17=request.form.get('City_C17'),
    City_C18=request.form.get('City_C18'),
    City_C19=request.form.get('City_C19'),
    City_C2=request.form.get('City_C2'),
    City_C20=request.form.get('City_C20'),
    City_C21=request.form.get('City_C21'),
    City_C22=request.form.get('City_C22'),
    City_C23=request.form.get('City_C23'),
    City_C24=request.form.get('City_C24'),
    City_C25=request.form.get('City_C25'),
    City_C26=request.form.get('City_C26'),
    City_C27=request.form.get('City_C27'),
    City_C28=request.form.get('City_C28'),
    City_C29=request.form.get('City_C29'),
    City_C3=request.form.get('City_C3'),
    City_C4=request.form.get('City_C4'),
    City_C5=request.form.get('City_C5'),
    City_C6=request.form.get('City_C6'),
    City_C7=request.form.get('City_C7'),
    City_C8=request.form.get('City_C8'),
    City_C9=request.form.get('City_C9')
)

        
        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        if results == 1:
            results="The driver will leave the company."
        else:
            results = "The driver will not leave the company"
        return render_template('home.html',results=results)

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080)        
    # app.run(host='0.0.0.0',port=5000, debug=True)#port=8080)
