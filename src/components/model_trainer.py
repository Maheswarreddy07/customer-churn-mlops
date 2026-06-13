import yaml
import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelTrainer:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.train_path = self.config["artifacts"]["train_data_file"]
        self.test_path = self.config["artifacts"]["test_data_file"]
        
        mlflow.set_tracking_uri(self.config["model_tracking"]["mlflow_tracking_uri"])
        mlflow.set_experiment(self.config["model_tracking"]["experiment_name"])

    def initiate_model_training(self):
        print("🏋️ Starting Model Training...")
        
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        
        X_train = train_df.drop(columns=["Churn"])
        y_train = train_df["Churn"]
        X_test = test_df.drop(columns=["Churn"])
        y_test = test_df["Churn"]
        
        params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": 42
        }
        
        with mlflow.start_run():
            model = XGBClassifier(**params)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred)
            }
            
            print(f"📊 Model Performance: {metrics}")
            
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            
            mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path="model",
                registered_model_name="XGBoost_Churn_Model"
            )
            
            print("🏆 Model successfully tracked and registered in MLflow Registry.")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.initiate_model_training()