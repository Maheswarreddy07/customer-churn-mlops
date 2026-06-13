import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import numpy as np

app=FastAPI(title="Customer Churn Prediction API",version="1.0")

class CustomerData(BaseModel):
    Age:float
    SubscriptionPlan:int
    MonthlyCharges:float
    ContractType:int
    SupportTickets:float

mlflow.set_tracking_uri("http://127.0.0.1:5000")

try:
    model_uri="models:/XGBoost_Churn_Model/latest"
    model=mlflow.pyfunc.load_model(model_uri)
    print("📡 Connected to ML flow Registry:Latest model loaded successfully.")
except Exception as e:
    print("⚠️ Could not reach MLflow registry. Ensure mlflow server is runningor model is registered.")
    model=None

@app.get("/")
def home():
    return {"message":"Customer Churn Prediction System API is Online"}

@app.post("/predict")
def predict_churn(data:CustomerData):
    if model is None:
        return {"error":"Model serving layer is unavailabel.Check system logs."}

    input_features=np.array([[
        data.Age,
        data.SubscriptionPlan,
        data.MonthlyCharges,
        data.ContractType,
        data.SupportTickets
    ]])

    predictions = model.predict(input_features)
    
    if isinstance(predictions, np.ndarray) and predictions.ndim > 1:
        churn_prob = float(predictions[0][1])  # Standard 2D array output
    else:
        churn_prob = float(predictions[0])     # Direct score/probability output
        
    prediction = int(churn_prob > 0.5)

    return{
        "churn_prediction":prediction,
        "churn_probability":round(churn_prob,2),
        "status":"High Risk" if prediction==1 else "Low Risk"
            }

if __name__=="__main__":
    uvicorn.run("app:app",host="0.0.0.0",port=8000,reload=True)