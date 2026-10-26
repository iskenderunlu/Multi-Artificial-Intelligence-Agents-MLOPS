# 🚀 Multi-Agent MLOps System

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.11-green.svg)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-1.4-orange.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

An autonomous, end-to-end **Multi-Agent MLOps (Machine Learning Operations)** system built entirely using 100% free and open-source tools (*Scikit-Learn, MLflow, Python*).

This project manages the entire machine learning lifecycle—from data preprocessing and model training to evaluation, tracking, and production deployment—without requiring manual intervention.

---

## 🤖 Multi-Agent Architecture

The system consists of 4 specialized autonomous agents:

1. **`DataPrepAgent` (Data Preparation Agent):**
   - Automatically checks and creates missing data directories (`data/`).
   - Preprocesses data, handles feature/target separation, and performs train-test splitting.
2. **`ModelTrainerAgent` (Model Training Agent):**
   - Autonomously trains multiple machine learning algorithms (*RandomForest*, *GradientBoosting*).
3. **`EvaluatorAgent` (Evaluation & Tracking Agent):**
   - Calculates performance metrics (`Accuracy`, `F1-Score`).
   - Logs experiments, metrics, and hyperparameter details directly into the **MLflow** registry.
   - Enforces automated quality checks against a defined accuracy threshold (90%).
4. **`DeployerAgent` (Deployment Agent):**
   - Safely exports the best-performing model that passes quality checks into `models/production_model.pkl` for live deployment.

---

## 📂 Project Structure

```text
multi_agent_mlops_project/
├── data/                    # Dataset directory created automatically
├── models/                  # Exported production models (.pkl)
├── src/
│   └── agents.py            # Core logic and classes for all 4 agents
├── main.py                  # Pipeline execution script
├── requirements.txt         # Project dependencies
├── .gitignore               # Excluded runtime files
└── README.md                # Project documentation
