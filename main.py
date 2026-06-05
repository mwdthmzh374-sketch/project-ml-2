
from fastapi import FastAPI
import pandas as pd
import joblib

app=FastAPI(title='Mobile Sales Prediction API')

model=joblib.load('models/best_model.pkl')

@app.get('/')
def home():
    return {'status':'running'}

@app.post('/predict')
def predict(data:dict):
    df=pd.DataFrame([data])
    prediction=float(model.predict(df)[0])
    return {'prediction':prediction}
