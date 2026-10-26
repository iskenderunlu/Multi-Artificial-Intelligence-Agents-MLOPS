import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

class DataPrepAgent:
    def __init__(self, data_path="data/dataset.csv"):
        self.data_path = data_path

    def run(self):
        print("🤖 [DataPrepAgent] Loading and preparing data...")
        
        # Ensure 'data' directory exists
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        
        # Sample dataset: Iris
        iris = load_iris(as_frame=True)
        df = iris.frame
        df['target'] = iris.target
        
        # Save dataset
        df.to_csv(self.data_path, index=False)
        
        X = df.drop(columns=['target'])
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("✅ [DataPrepAgent] Data successfully split: Train %d, Test %d" % (len(X_train), len(X_test)))
        return X_train, X_test, y_train, y_test

class ModelTrainerAgent:
    def __init__(self):
        pass

    def train_models(self, X_train, y_train):
        print("🤖 [ModelTrainerAgent] Training models...")
        models = {
            "RandomForest": RandomForestClassifier(n_estimators=50, random_state=42),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=50, random_state=42)
        }
        
        trained_models = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            trained_models[name] = model
            print(f"  └─ {name} trained.")
        
        return trained_models

class EvaluatorAgent:
    def __init__(self, metric_threshold=0.90):
        self.threshold = metric_threshold

    def evaluate_and_log(self, models, X_test, y_test):
        print("🤖 [EvaluatorAgent] Evaluating models and logging to MLflow...")
        best_model = None
        best_accuracy = 0.0
        best_name = ""

        mlflow.set_experiment("Multi_Agent_MLOps_Pipeline")

        for name, model in models.items():
            with mlflow.start_run(run_name=name):
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds, average='weighted')

                # MLflow Logging
                mlflow.log_param("model_type", name)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1_score", f1)
                
                # Trust Tree types for scikit-learn models
                mlflow.sklearn.log_model(
                    model, 
                    artifact_path="model", 
                    skops_trusted_types=["sklearn.tree._tree.Tree"]
                )

                print(f"  └─ {name} -> Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

                if acc > best_accuracy:
                    best_accuracy = acc
                    best_model = model
                    best_name = name

        # Threshold check
        approved = best_accuracy >= self.threshold
        print(f"🏆 [EvaluatorAgent] Best Model: {best_name} (Accuracy: {best_accuracy:.4f})")
        if approved:
            print("✅ [EvaluatorAgent] Model passed quality threshold (Approved for Deployment).")
        else:
            print("❌ [EvaluatorAgent] Model failed quality threshold.")

        return best_name, best_model, approved

class DeployerAgent:
    def __init__(self, export_path="models/production_model.pkl"):
        self.export_path = export_path

    def deploy(self, model_name, model, is_approved):
        print("🤖 [DeployerAgent] Starting deployment process...")
        if not is_approved:
            print("⚠️ [DeployerAgent] Model deployment CANCELLED due to lack of approval.")
            return False

        # Ensure 'models' directory exists
        os.makedirs(os.path.dirname(self.export_path), exist_ok=True)

        # Export model to production directory
        joblib.dump(model, self.export_path)
        print(f"🚀 [DeployerAgent] {model_name} successfully saved to '{self.export_path}' and deployed to production!")
        return True