# Use a lightweight Python base image
FROM python:3.11-slim

# Prevent Python from writing bytecode and buffering output
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose default Streamlit port
EXPOSE 8501

# Run Streamlit dynamically binding to Back4app's allocated $PORT
CMD ["sh", "-c", "streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false"]