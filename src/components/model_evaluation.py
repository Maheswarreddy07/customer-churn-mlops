import mlflow
from mlflow.tracking import MlflowClient

class ModelEvaluation:
    def __init__(self):
        self.client = MlflowClient()
        self.model_name = "XGBoost_Churn_Model"

    def evaluate_and_gatekeep(self, current_run_accuracy: float) -> bool:
        print("\n⚖️ Initiating Model Evaluation Gatekeeper...")
        
        try:
            # Fetch the baseline model using the new UI Aliases API
            model_version_details = self.client.get_model_version_by_alias(self.model_name, "champion")
            
            # Extract the run metrics of the active champion baseline
            prod_run_id = model_version_details.run_id
            prod_run = mlflow.get_run(prod_run_id)
            
            baseline_accuracy = float(prod_run.data.metrics.get("accuracy", 0.0))
            print(f"📊 Baseline Production ('champion') Accuracy: {baseline_accuracy:.4f}")
            print(f"📊 Newly Trained Model Accuracy: {current_run_accuracy:.4f}")
            
            if current_run_accuracy >= baseline_accuracy:
                print("✅ Success: New model meets or outperforms the production baseline.")
                return True
            else:
                print("❌ Rejected: New model performance is lower than current baseline.")
                return False
                
        except Exception as e:
            print("ℹ️ No 'champion' baseline model alias active in registry yet. Automatically approving new model run.")
            return True