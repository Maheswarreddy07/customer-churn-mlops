Customer Churn MLOps Pipeline

A modular, real-world machine learning system that takes messy raw customer data, cleans it up, trains a model, tracks its performance, and hosts it behind an interactive web dashboard. Built the way real ML teams build software — config-driven, split into clean stages, and backed by automated tests.


What this project does

This system sets up a fully automated loop to predict which customers are about to churn:


Data Pipeline — Reads raw customer data, flags missing values or out-of-bounds inputs (like negative ages), cleans formatting, and prepares it for training.
Model Training — Trains an XGBoost classifier and logs parameters, metrics, and artifacts to MLflow.
Model Gatekeeper — Before a new model is saved, compares its accuracy against the current production model in the MLflow registry. Worse models are blocked from deployment.
Live API — A FastAPI backend that pulls the current "champion" model directly from the MLflow registry at runtime — no hardcoded model files.
Dashboard — A Streamlit web app where anyone can enter a customer's age, monthly charges, and support ticket count and instantly see a churn risk score with a recommended action.



Project structure

customer-churn-mlops/
│
├── .github/
│   Continuous integration setup
│   │
│   └── workflows/
│       │
│       └── test_pipeline.yml
│           CI script — runs tests automatically on every push
│
├── config/
│   Centralized configuration
│   │
│   └── config.yaml
│       Single source of truth for file paths, schema, and model settings
│
├── data/
│   Dataset storage
│   │
│   ├── raw/
│   │   Fresh, untouched CSV datasets
│   │
│   └── processed/
│       Cleaned, scaled, numeric data ready for training
│
├── src/
│   Core application code
│   │
│   ├── __init__.py
│   │   Marks this directory as a Python package
│   │
│   ├── components/
│   │   The building blocks of the pipeline
│   │   │
│   │   ├── data_ingestion.py
│   │   │   Loads raw files and splits data into train/test sets
│   │   │
│   │   ├── data_validation.py
│   │   │   Checks data integrity and flags broken or invalid records
│   │   │
│   │   ├── data_transformation.py
│   │   │   Imputes missing values, encodes categories, and scales inputs
│   │   │
│   │   ├── model_trainer.py
│   │   │   Trains the XGBoost model and logs metrics to MLflow
│   │   │
│   │   └── model_evaluation.py
│   │       Compares newly trained models against the production baseline
│   │
│   ├── pipeline/
│   │   │
│   │   └── training_pipeline.py
│   │       The conductor — runs all components in order
│   │
│   └── api/
│       │
│       ├── app.py
│       │   FastAPI backend that serves real-time predictions
│       │
│       └── dashboard.py
│           Streamlit frontend for interacting with the model
│
├── tests/
│   Safety nets
│   │
│   ├── conftest.py
│   │   Test setup and shared path configuration
│   │
│   └── test_components.py
│       Throws bad data at the pipeline to verify it holds up
│
├── Dockerfile
│   Bundles the application into a container
│
├── requirements.txt
│   Pinned Python dependencies
│
└── README.md


How to run this locally

1. Boot up the MLflow Tracking Server

Keep this window active to handle metric tracking and the model registry:

bashmlflow server --host 127.0.0.1 --port 5000

2. Run the training pipeline

bashpython -m src.pipeline.training_pipeline

This runs ingestion → validation → transformation → training → evaluation, end to end.

3. Start the API

bashuvicorn src.api.app:app --reload

4. Launch the dashboard

bashstreamlit run src/api/dashboard.py

Or run everything in Docker

bashdocker build -t churn-mlops .
docker run -p 8000:8000 churn-mlops


Running tests

bashpytest tests/

Tests run automatically on every push via GitHub Actions (.github/workflows/test_pipeline.yml).


Tech stack

Pandas, NumPy, Scikit-learn, XGBoost, MLflow, Evidently, FastAPI, Pydantic, Uvicorn, Streamlit, Pytest, Docker