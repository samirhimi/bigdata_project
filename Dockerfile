# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.12-slim AS builder 

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create a non-root user

WORKDIR /app

# RUN chown -R devops:devops /home/devops/app

COPY ./app .

# Install pip requirements

RUN python -m pip install --no-cache-dir -r requirements.txt &&  \
    python -m pip install --no-cache-dir pandas

RUN  python alibaba-trace-ML-Compare.py > /app/data.txt

FROM python:3.12-slim 

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd -r devops && useradd -r -g devops -m devops

COPY --from=builder /app/data.txt /home/devops/app/data.txt

USER  devops

EXPOSE 8080

WORKDIR /home/devops/app/

CMD [ "python3", "-m", "http.server", "8080"]