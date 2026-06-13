import pytest
import pandas as pd
import os
from src.components.data_validation import DataValidation

def test_data_validation_logic():
    """Test if our validation script catches structural anomalies."""
    validator = DataValidation()
    
    # Create a temporary 'bad' dataset
    bad_data = {
        "CustomerID": ["CUST-TEST"],
        "Age": [-50],  # Invalid age
        "SubscriptionPlan": ["Premium"],
        "MonthlyCharges": [100.0],
        "ContractType": ["One year"],
        "SupportTickets": [2],
        "Churn": [0]
    }
    
    df = pd.DataFrame(bad_data)
    temp_file = "data/raw/temp_test_data.csv"
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(temp_file, index=False)
    
    # Run validator
    status = validator.validate_data(temp_file)
    
    # Clean up test file
    if os.path.exists(temp_file):
        os.remove(temp_file)
        
    assert status is True