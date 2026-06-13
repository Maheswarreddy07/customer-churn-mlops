import pytest
import pandas as pd
import os
import numpy as np
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation

def test_edge_case_missing_and_corrupted_values():
    """Edge Case 1: Testing how components handle empty/NaN cells and negative data fields."""
    validator = DataValidation()
    
    # Intentionally messy data: Missing SubscriptionPlan, negative charges, missing CustomerID
    corrupted_data = {
        "CustomerID": [np.nan, "CUST-002"],
        "Age": [45, -10],                     # Negative age outlier
        "SubscriptionPlan": [1, np.nan],       # NaN value in categorical data
        "MonthlyCharges": [-99.0, 120.5],      # Negative monetary value
        "ContractType": [0, 1],
        "SupportTickets": [np.nan, 4],         # Missing numeric values
        "Churn": [0, 1]
    }
    
    df = pd.DataFrame(corrupted_data)
    edge_file = "data/raw/edge_case_test_data.csv"
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(edge_file, index=False)
    
    # Ensure validation captures structural shapes without a hard crash
    status = validator.validate_data(edge_file)
    
    if os.path.exists(edge_file):
        os.remove(edge_file)
        
    assert status is True  # Core structure holds up, validation catches the errors internally

def test_edge_case_extreme_values():
    """Edge Case 2: Testing how data fields handle extreme scale/unseen data values."""
    validator = DataValidation()
    
    # Data with cartoonishly extreme inputs
    extreme_data = {
        "CustomerID": ["CUST-MAX"],
        "Age": [999],                          # Century-old user
        "SubscriptionPlan": [99999],           # Unseen category code
        "MonthlyCharges": [999999.99],         # Massive float input
        "ContractType": [0],
        "SupportTickets": [500],               # Massive scale integer
        "Churn": [0]
    }
    
    df = pd.DataFrame(extreme_data)
    extreme_file = "data/raw/extreme_test_data.csv"
    df.to_csv(extreme_file, index=False)
    
    status = validator.validate_data(extreme_file)
    
    if os.path.exists(extreme_file):
        os.remove(extreme_file)
        
    assert status is True