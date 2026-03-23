import sys
import mlflow
import os

def main():
    try:
        with open("model_info.txt", "r") as f:
            run_id = f.read().strip()
    except FileNotFoundError:
        print("Error: model_info.txt not found. Did the validation job export it?")
        sys.exit(1)
    
    print(f"Checking accuracy for Run ID: {run_id}")
    
    # We allow a local fallback just in case the CI runs without MLFLOW_TRACKING_URI set.
    # Otherwise MLflow throws exceptions trying to query local sqlite vs remote.
    # But ideally MLFLOW_TRACKING_URI handles it all automatically.
    try:
        run = mlflow.get_run(run_id)
        accuracy = run.data.metrics.get("final_d_accuracy", 0.0)
    except Exception as e:
        print(f"Error retrieving run from MLflow (is the URI correct?): {e}")
        # If we really just want to simulate a pass/fail natively if no DagsHub is attached
        sys.exit(1)
        
    print(f"Accuracy retrieved: {accuracy}")
    
    if accuracy < 0.85:
        print(f"Validation failed! Accuracy {accuracy} is below threshold 0.85")
        sys.exit(1)
    else:
        print(f"Validation passed! Accuracy {accuracy} is strictly acceptable.")

if __name__ == "__main__":
    main()
