# Use lightweight official Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Command to run the FastAPI server on port 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]