# Use a lightweight official Python base image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer outputs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set container working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy serialized model artifacts and source code
COPY models/ ./models/
COPY src/ ./src/

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI server binding to 0.0.0.0 inside container
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]