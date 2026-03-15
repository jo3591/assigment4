# Assignment 4 - ML Pipeline CI

## Public GitHub Repository
The pipeline is hosted securely in this repository: [https://github.com/jo3591/assigment4](https://github.com/jo3591/assigment4)

## Explanation of the CI Pipeline & Green Build
The GitHub Actions pipeline `.github/workflows/ml-pipeline.yml` automates the Continuous Integration strategy for our Machine Learning model. 

1. **Trigger Condition**: The workflow triggers immediately upon a `push` to any branch **except** `main`, as well as on any `pull_request`. This ensures proposed changes are validated before merging.
2. **Setup**: The environment is scaffolded on an `ubuntu-latest` runner by fetching the code repository (`actions/checkout@v4`) and provisioning Python 3.10 (`actions/setup-python@v5`).
3. **Dependencies**: `pip install -r requirements.txt` fetches all core libraries (numpy, pandas, tensorflow, keras, mlflow). 
4. **Linter Check**: Validation with `flake8` is performed. The linter scans for major syntax and styling errors.
5. **Model Dry Test**: A quick verification (`python -c "import tensorflow; print('Model environment ready!')"`) actively ensures that the primary DL framework (TensorFlow) loaded correctly in the job container.
6. **Artifacting**: Finally, `actions/upload-artifact@v4` successfully captures this `README.md` document and generates a traceable artifact named `project-doc`.

Once committed, a "Green Check" reflects the holistic, error-free completion of all these stages, demonstrating robust code health and integration readiness.

## Author 
Yousef Hendy
