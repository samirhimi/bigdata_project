# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.12-slim

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1


# Install pip requirements

WORKDIR /app

COPY ./app /app

RUN python -m pip install -r --no-cache-dir requirements.txt 

EXPOSE 8080

RUN  python alibaba-trace-ML-Compare.py | tee data.txt

# CMD [ "python3", "-m", "http.server", "8080"]

