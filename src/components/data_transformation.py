import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder

class DataTransformation:
    def __init__(self,config_path="config/config.yaml"):
        with open(config_path,"r") as f:
            self.config=yaml.safe_load(f)

        self.raw_data_path=self.config["artifacts"]["raw_data_file"]
        self.train_data_path=self.config["artifacts"]["train_data_file"]
        self.test_data_path=self.config["artifacts"]["test_data_file"]
        self.processed_dir=self.config["artifacts"]["processed_data_dir"]

    def initiate_data_transformation(self):
        print("🔄 Starting Data Transformation...")
        df=pd.read_csv(self.raw_data_path)

        df=df[df["Age"]>=self.config["data_validation"]["min_age"]]

        df["MonthlyCharges"]=df["MonthlyCharges"].fillna(df["MonthlyCharges"].median())

        df=df.drop(columns=["CustomerID"])

        categorical_cols=["SubscriptionPlan","ContractType"]
        for col in categorical_cols:
            le=LabelEncoder()
            df[col]=le.fit_transform(df[col])

        X=df.drop(columns=["Churn"])
        y=df["Churn"]

        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


        scalar=StandardScaler()
        X_train_scaled=scalar.fit_transform(X_train)
        X_test_scaled=scalar.transform(X_test)

        train_df=pd.DataFrame(X_train_scaled,columns=X.columns)
        train_df["Churn"]=y_train.values

        test_df=pd.DataFrame(X_test_scaled,columns=X.columns)
        test_df["Churn"]=y_test.values
   
        os.makedirs(self.processed_dir,exist_ok=True)

        train_df.to_csv(self.train_data_path,index=False)
        test_df.to_csv(self.test_data_path,index=False)

        print(f"✅ Data transformed and saved:\n Train:{self.train_data_path}\n Test:{self.test_data_path}")
        return self.train_data_path,self.test_data_path

if __name__=="__main__":
    transformer=DataTransformation()
    transformer.initiate_data_transformation()

        