# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.12-slim AS builder

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create a non-root user
RUN groupadd -r devops && useradd -r -g devops devops

WORKDIR /app

RUN chown -R devops:devops /app

COPY ./app .

# Install pip requirements

RUN python -m pip install --no-cache-dir --user -r requirements.txt 

USER  devops

EXPOSE 8080

RUN  python alibaba-trace-ML-Compare.py > /home/data.txt

CMD [ "python3", "-m", "http.server", "8080", "--directory", "/home/data.txt"]

