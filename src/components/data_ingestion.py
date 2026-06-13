import os
import yaml
import pandas as pd
import numpy as np

class DataIngestion:
    def __init__(self,config_path="config/config.yaml"):
        with open(config_path,"r") as f:
            self.config=yaml.safe_load(f)

        self.raw_data_path=self.config["artifacts"]["raw_data_file"]
        self.raw_dir=self.config["artifacts"]["raw_data_dir"]
    
    def create_mock_data(self, num_records=1000):
        os.makedirs(self.raw_dir, exist_ok=True)

        np.random.seed(42)
        data={
            "CustomerID":[f"CUST-{i:04d}" for i in range(num_records)],
            "Age":np.random.randint(18,80,size=num_records).tolist(),
            "SubscriptionPlan":np.random.choice(["Basic","Standard","Premium"], size=num_records).tolist(),
            "MonthlyCharges":np.round(np.random.uniform(20,150,size=num_records),2).tolist(),
            "ContractType":np.random.choice(["Month-to-month", "One year", "Two year"],size=num_records).tolist(),
            "SupportTickets":np.random.randint(0,10,size=num_records).tolist(),
            "Churn":np.random.choice([0,1],p=[0.75,0.25],size=num_records).tolist()
        }

        df=pd.DataFrame(data)
        df.loc[0,"Age"]=-5
        df.loc[1,"MonthlyCharges"]=np.nan

        df.to_csv(self.raw_data_path,index=False)
        print("📦 Mock raw data generated and saved to:{self.raw_data_path}")

    def initiate_data_ingestion(self):
        print("🚀 Starting Data Ingestion...")
        if not os.path.exists(self.raw_data_path):
            self.create_mock_data()

        df=pd.read_csv(self.raw_data_path)
        return self.raw_data_path

if __name__=="__main__":
    ingestion=DataIngestion()
    ingestion.initiate_data_ingestion()
