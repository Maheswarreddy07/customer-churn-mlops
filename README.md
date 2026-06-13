Customer Churn MLOps Pipeline: From Raw Data to Live UI

A modular, real-world machine learning system that takes messy raw customer data, cleans it up, trains a model, tracks its performance, and hosts it behind an interactive web dashboard.

Instead of cramming everything into one messy notebook, this project is structured the way real ML teams build software: config-driven, split into clean steps, and backed by automated tests that catch breaking errors before they ship.


What does this project actually do?

Imagine a company wants to predict which customers are about to cancel their subscriptions. This system sets up a fully automated loop to solve that problem:


The Core Pipeline (src/pipeline/) – Reads fresh raw data, flags missing values or out-of-bounds inputs (like negative ages), cleans up formatting, and trains an XGBoost model.
The Smart Gatekeeper (src/components/model_evaluation.py) – Before a newly trained model is saved, this step connects to the MLflow registry and compares the new model's accuracy against the current production model. If the new one is worse, the deployment is blocked.
The Live API (src/api/app.py) – A FastAPI backend that doesn't read a hardcoded model file from disk. Instead, it pulls the current "champion" model directly from the MLflow registry at runtime.
The User Dashboard (src/api/dashboard.py) – A Streamlit web app with simple sliders and forms. Anyone on the business side can enter a customer's age, monthly charges, and support ticket count and instantly see a churn risk score with a recommended action.



Project layout

customer-churn-mlops/
│
├── .github/workflows/
│   └── test_pipeline.yml        # CI script — runs tests automatically on every push
│
├── config/
│   └── config.yaml              # Single source of truth for paths, schema, and model settings
│
├── data/
│   ├── raw/                      # Fresh, untouched CSV datasets
│   └── processed/                # Cleaned, scaled, numeric data ready for training
│
├── src/
│   ├── __init__.py
│   │
│   ├── components/
│   │   ├── data_ingestion.py        # Loads files, splits into train/test
│   │   ├── data_validation.py       # Checks data integrity, flags broken records
│   │   ├── data_transformation.py   # Imputes missing values, scales inputs
│   │   ├── model_trainer.py         # Trains the XGBoost model, logs metrics to MLflow
│   │   └── model_evaluation.py      # Compares new models against the production baseline
│   │
│   ├── pipeline/
│   │   └── training_pipeline.py     # Conductor — runs all components in order
│   │
│   └── api/
│       ├── app.py                   # FastAPI backend (serves predictions)
│       └── dashboard.py             # Streamlit frontend (user interface)
│
├── tests/
│   ├── conftest.py                  # Test setup and path config
│   └── test_components.py           # Throws bad data at the pipeline to check it holds up
│
├── Dockerfile                       # Bundles the app into a container
├── requirements.txt                 # Pinned Python dependencies
└── README.md


How to run it

1. Run the training pipeline

bashpython -m src.pipeline.training_pipeline

This runs ingestion → validation → transformation → training → evaluation, end to end.

2. Start MLflow

Keep this terminal window open — it handles metric tracking and the model registry:

bashmlflow server --host 127.0.0.1 --port 5000

3. Start the API

bashuvicorn src.api.app:app --reload

4. Start the dashboard

bashstreamlit run src/api/dashboard.py

Or run everything in Docker:

bashdocker build -t churn-mlops .
docker run -p 8000:8000 churn-mlops


Running tests

bashpytest tests/

Tests run automatically on every push via GitHub Actions (.github/workflows/test_pipeline.yml).


Tech stack

Pandas, NumPy, Scikit-learn, XGBoost, MLflow, Evidently, FastAPI, Pydantic, Uvicorn, Streamlit, Pytest, Docker