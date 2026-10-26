import os
from src.agents import DataPrepAgent, ModelTrainerAgent, EvaluatorAgent, DeployerAgent

def run_pipeline():
    print("==================================================")
    print("🚀 STARTING MULTI-AGENT MLOPS PIPELINE")
    print("==================================================")

    # Initialize Agents
    data_agent = DataPrepAgent()
    trainer_agent = ModelTrainerAgent()
    evaluator_agent = EvaluatorAgent(metric_threshold=0.90)
    deployer_agent = DeployerAgent()

    # Step 1: Data Preparation
    X_train, X_test, y_train, y_test = data_agent.run()
    print("-" * 50)

    # Step 2: Model Training
    models = trainer_agent.train_models(X_train, y_train)
    print("-" * 50)

    # Step 3: Evaluation & Logging
    best_name, best_model, approved = evaluator_agent.evaluate_and_log(models, X_test, y_test)
    print("-" * 50)

    # Step 4: Deployment
    deployer_agent.deploy(best_name, best_model, approved)

    print("==================================================")
    print("🎉 MLOPS PIPELINE COMPLETED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_pipeline()