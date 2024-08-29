from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

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
        data=CustomData(   
            Company=request.form.get('Company'),
            TypeName=request.form.get('TypeName'),
            OpSys=request.form.get('OpSys'),
            Ram=float(request.form.get('Ram')),
            Weight=float(request.form.get('Weight')),
            Touchscreen=float(request.form.get('Touchscreen')),
            Ips=float(request.form.get('Ips')),
            ppi=float(request.form.get('ppi')),
            Cpu_Brand=request.form.get('Cpu_Brand'), 
            Gpu_Brand=request.form.get('Gpu_Brand'),
            HDD=request.form.get('HDD'),
            SSD=request.form.get('SSD'),

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    
    
if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)

    




