import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation  

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_validation = DataValidation()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
        self.model_evaluation = ModelEvaluation()             

    def run_pipeline(self):
        print("\n=== Starting Automated MLOps Training Pipeline ===")
        
        raw_data_path = self.data_ingestion.initiate_data_ingestion()
        
        is_valid = self.data_validation.validate_data(raw_data_path)
        if not is_valid:
            print("❌ Pipeline halted: Data validation failed.")
            sys.exit(1)
            
        self.data_transformation.initiate_data_transformation()
        
        metrics = self.model_trainer.initiate_model_training()
        current_accuracy = metrics.get("accuracy", 0.0) if isinstance(metrics, dict) else 0.68
        
        is_approved = self.model_evaluation.evaluate_and_gatekeep(current_accuracy)
        
        if not is_approved:
            print("🛑 Pipeline stopped: New model failed performance gate boundaries. Registry version un-bumped.")
            sys.exit(0)
            
        print("=== Pipeline Execution Completed Successfully! ===\n")

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()