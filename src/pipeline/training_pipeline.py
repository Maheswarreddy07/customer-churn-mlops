import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion=DataIngestion()
        self.data_validation=DataValidation()
        self.data_transformation=DataTransformation()
        self.model_trainer=ModelTrainer()

    def run_pipeline(self):
        print("n Starting Automated MLOps Training Pipeline")

        raw_data_path=self.data_ingestion.initiate_data_ingestion()

        is_valid=self.data_validation.validate_data(raw_data_path)
        if not is_valid:
            print("❌ Pipeline halted:Data validation failed.")
            sys.exit(1)

        self.data_transformation.initiate_data_transformation()

        self.model_trainer.initiate_model_training()

        print("Pipeline Execution Completed Successfully!\n")

if __name__=="__main__":
    pipeline=TrainingPipeline()
    pipeline.run_pipeline()