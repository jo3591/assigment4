import sys
import mlflow
import os


def main():
    try:
        with open("model_info.txt", "r") as f:
            lines = f.read().strip().split("\n")
            run_id       = lines[0]
            fallback_acc = float(lines[1]) if len(lines) > 1 else 0.0
    except FileNotFoundError:
        print("Error: model_info.txt not found. Did the validate job upload it?")
        sys.exit(1)

    print(f"Checking accuracy for Run ID: {run_id}")

    accuracy = 0.0
    try:
        run      = mlflow.get_run(run_id)
        accuracy = run.data.metrics.get("final_d_accuracy", 0.0)
        print(f"Accuracy retrieved from MLflow: {accuracy}")
    except Exception as e:
        print(f"Could not reach MLflow server: {e}")
        print(f"Falling back to accuracy stored in model_info.txt: {fallback_acc}")
        accuracy = fallback_acc

    THRESHOLD = 0.99
    if accuracy < THRESHOLD:
        print(f"FAILED — accuracy {accuracy:.4f} is below threshold {THRESHOLD}")
        sys.exit(1)
    else:
        print(f"PASSED — accuracy {accuracy:.4f} meets threshold {THRESHOLD}")


if __name__ == "__main__":
    main()