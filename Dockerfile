# This is an example Dockerfile that builds a minimal container for running LK Agents
# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12.1
FROM python:${PYTHON_VERSION}-slim

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Create a non-privileged user that the app will run under.
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/appuser" \
    --shell "/sbin/nologin" \
    --uid "10001" \
    appuser

# Install gcc and other build dependencies.
RUN apt-get update && \
    apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up the application directory
WORKDIR /home/appuser

RUN mkdir -p /home/appuser/.cache && \
    chown -R appuser:appuser /home/appuser

# Switch to non-root user
USER appuser

# Copy requirements first for better caching
COPY requirements.txt .
RUN python -m pip install --user --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# ensure that any dependent models are downloaded at build-time
RUN python agent.py download-files

# expose healthcheck port
EXPOSE 8081

# Run the application.
CMD ["python", "agent.py", "dev"]