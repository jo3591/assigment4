FROM python:3.10-slim

ARG RUN_ID

# Include a command to "download" the model
RUN echo "Downloading model ${RUN_ID}"

CMD ["echo", "Container ready!"]
