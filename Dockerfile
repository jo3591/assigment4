FROM python:3.10-slim

# Accept the MLflow Run ID as a build argument
ARG RUN_ID

# Simulate downloading the model artifact for this run
RUN echo "Downloading model for MLflow Run ID: ${RUN_ID}"

CMD ["echo", "Container is ready!"]