from typing import Union
import joblib
from fastapi import FastAPI
import numpy as np
import pandas as pd
from models.Animal import Animal

joblib_in = open("./gradboost_model_v1.joblib","rb")
model=joblib.load(joblib_in)
app = FastAPI()

key_map = {
    0:'grazing',
    1:'running',
    2:'standing',
    3:'trotting',
    4:'walking'
}

animal_map = {
    0:'Sheep',
    1:'Goat'
}

@app.get('/')
def index():
    return {'message': 'AI Cattle Monitor'}

@app.post("/cattle-predict")
def predict_segment(data:Animal):
    data = data.dict()
    ax=data['ax']
    ay=data['ay']
    az=data['az']

    cx,cy,cz = data['cx'],data['cy'],data['cz']
    axhg,ayhg,azhg = data['axhg'],data['ayhg'],data['azhg']
    acc = np.sqrt((ax**2)+(ay**2)+(az**2))
    hacc = np.sqrt((axhg**2)+(ayhg**2)+(azhg**2))

    datapoint = pd.DataFrame({'animal_type':data['animal_type'],'ax_avg':ax,'ay_avg':ay,'az_avg':az,'acc_avg':acc,'hgacc_avg':hacc,'axhg_avg':axhg,'ayhg_avg':ayhg,'azhg_avg':azhg,'cx_avg':cx,'cy_avg':cy,'cz_avg':cz}, index=[0])

    prediction = model.predict(datapoint[0:1])

    
    return {
        'animal_type':animal_map[data['animal_type']],
        'prediction': key_map[prediction[0]]
    }