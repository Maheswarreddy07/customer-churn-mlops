import yaml
import pandas as pd

class DataValidation:
    def __init__(self,config_path="config/config.yaml"):
        with open(config_path,"r") as f:
            self.config=yaml.safe_load(f)

        self.validation_config=self.config["data_validation"]

    def validate_data(self,file_path:str)->bool:
        print("🔍 Starting Data Validation...")
        try:
            df=pd.read_csv(file_path)
            validation_status=True

            expected_cols=self.validation_config["expected_columns"]
            for col in expected_cols:
                if col not in df.columns:
                    print(f"❌ Validation Failed: Missing column '{col}'")
                    validation_status=False

            null_counts=df.isnull().sum().sum()
            if null_counts>0:
                print(f"⚠️ Warning: Found {null_counts} missing values. Transformation step will handle them.")

            min_age=self.validation_config["min_age"]
            max_age=self.validation_config["max_age"]

            invalid_ages=df[(df["Age"]<min_age)|(df["Age"]>max_age)]
            if not invalid_ages.empty:
                print(f"⚠️ Warning: Detected {len(invalid_ages)} rows with out-of-bounds Ages (e.g., row index 0:{df.loc[0,'Age']}).Data cleaning will drop these.")

            if validation_status:
                print("✅ Data validation completed successfully.")
            return validation_status

        except Exception as e:
            print(f"❌ Validation failed completely due to exception:{str(e)}")
            return False

if __name__=="__main__":
    validator=DataValidation()
    validator.validate_data("data/raw/customer_churn.csv")